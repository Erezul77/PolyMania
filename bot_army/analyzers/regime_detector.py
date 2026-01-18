"""Market regime detection."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("analyzer.regime")


class MarketRegime(Enum):
    """Market regime types."""
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    MEAN_REVERTING = "MEAN_REVERTING"
    RANGING = "RANGING"
    CRISIS = "CRISIS"
    EUPHORIA = "EUPHORIA"
    PRE_EVENT = "PRE_EVENT"
    POST_EVENT = "POST_EVENT"


@dataclass
class RegimeState:
    """Current regime state."""
    primary: MarketRegime
    secondary: Optional[MarketRegime] = None
    confidence: float = 0.0
    volatility_percentile: float = 50.0
    trend_strength: float = 0.0
    mean_reversion_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary": self.primary.value,
            "secondary": self.secondary.value if self.secondary else None,
            "confidence": self.confidence,
            "volatility_percentile": self.volatility_percentile,
            "trend_strength": self.trend_strength,
            "mean_reversion_score": self.mean_reversion_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class RegimeDetector:
    """
    Detects market regime for strategy adaptation.
    Uses statistical analysis to classify market conditions.
    """
    
    # Regime-specific strategy recommendations
    REGIME_STRATEGIES = {
        MarketRegime.HIGH_VOLATILITY: {
            "position_size": 0.5,  # Reduce size
            "stop_loss": 1.5,  # Wider stops
            "take_profit": 2.0,  # Larger targets
            "strategy": "momentum"
        },
        MarketRegime.LOW_VOLATILITY: {
            "position_size": 1.2,  # Increase size
            "stop_loss": 0.8,  # Tighter stops
            "take_profit": 0.8,  # Smaller targets
            "strategy": "mean_reversion"
        },
        MarketRegime.TRENDING_UP: {
            "position_size": 1.0,
            "stop_loss": 1.0,
            "take_profit": 1.5,
            "strategy": "trend_following",
            "bias": "long"
        },
        MarketRegime.TRENDING_DOWN: {
            "position_size": 1.0,
            "stop_loss": 1.0,
            "take_profit": 1.5,
            "strategy": "trend_following",
            "bias": "short"
        },
        MarketRegime.MEAN_REVERTING: {
            "position_size": 1.0,
            "stop_loss": 1.0,
            "take_profit": 1.0,
            "strategy": "mean_reversion"
        },
        MarketRegime.RANGING: {
            "position_size": 0.8,
            "stop_loss": 0.7,
            "take_profit": 0.7,
            "strategy": "range_trading"
        },
        MarketRegime.CRISIS: {
            "position_size": 0.3,  # Minimal exposure
            "stop_loss": 2.0,
            "take_profit": 3.0,
            "strategy": "defensive"
        },
        MarketRegime.EUPHORIA: {
            "position_size": 0.7,  # Cautious
            "stop_loss": 1.2,
            "take_profit": 1.5,
            "strategy": "contrarian"
        },
        MarketRegime.PRE_EVENT: {
            "position_size": 0.5,
            "stop_loss": 1.5,
            "take_profit": 2.0,
            "strategy": "event_driven"
        },
        MarketRegime.POST_EVENT: {
            "position_size": 0.8,
            "stop_loss": 1.0,
            "take_profit": 1.2,
            "strategy": "momentum"
        }
    }
    
    def __init__(self):
        self._regime_history: List[RegimeState] = []
        self._market_regimes: Dict[str, RegimeState] = {}
        self._history_size = 100
    
    def detect(
        self,
        market_id: str,
        prices: List[float],
        volumes: List[float] = None,
        event_time: datetime = None
    ) -> RegimeState:
        """Detect current market regime."""
        if len(prices) < 20:
            return RegimeState(
                primary=MarketRegime.RANGING,
                confidence=0.3
            )
        
        prices_arr = np.array(prices)
        
        # Calculate regime indicators
        volatility_score = self._calculate_volatility_regime(prices_arr)
        trend_score = self._calculate_trend_regime(prices_arr)
        mean_rev_score = self._calculate_mean_reversion(prices_arr)
        
        # Event-based regime
        event_regime = None
        if event_time:
            hours_to_event = (event_time - datetime.utcnow()).total_seconds() / 3600
            if 0 < hours_to_event < 24:
                event_regime = MarketRegime.PRE_EVENT
            elif -4 < hours_to_event <= 0:
                event_regime = MarketRegime.POST_EVENT
        
        # Determine primary regime
        primary_regime, confidence = self._classify_regime(
            volatility_score, trend_score, mean_rev_score
        )
        
        # Override with event regime if applicable
        if event_regime:
            secondary = primary_regime
            primary_regime = event_regime
        else:
            secondary = None
        
        state = RegimeState(
            primary=primary_regime,
            secondary=secondary,
            confidence=confidence,
            volatility_percentile=volatility_score * 100,
            trend_strength=trend_score,
            mean_reversion_score=mean_rev_score,
            metadata={
                "price_count": len(prices),
                "current_price": prices[-1],
                "sma_20": np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            }
        )
        
        # Store state
        self._market_regimes[market_id] = state
        self._regime_history.append(state)
        self._regime_history = self._regime_history[-self._history_size:]
        
        return state
    
    def _calculate_volatility_regime(self, prices: np.ndarray) -> float:
        """Calculate volatility regime score (0-1)."""
        returns = np.diff(prices) / prices[:-1]
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 10:
            return 0.5
        
        # Current volatility
        vol_short = np.std(returns[-10:])
        vol_long = np.std(returns)
        
        # Percentile
        percentile = vol_short / vol_long if vol_long > 0 else 1
        
        return min(1.0, max(0.0, percentile))
    
    def _calculate_trend_regime(self, prices: np.ndarray) -> float:
        """Calculate trend strength (-1 to 1)."""
        if len(prices) < 20:
            return 0.0
        
        # Linear regression
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize by price level
        normalized_slope = slope / np.mean(prices) * 100
        
        # ADX-like calculation
        returns = np.diff(prices) / prices[:-1]
        up_moves = sum(1 for r in returns if r > 0)
        down_moves = len(returns) - up_moves
        
        directional_ratio = (up_moves - down_moves) / len(returns) if returns.size > 0 else 0
        
        # Combine slope and direction
        trend_score = (normalized_slope + directional_ratio) / 2
        
        return max(-1.0, min(1.0, trend_score))
    
    def _calculate_mean_reversion(self, prices: np.ndarray) -> float:
        """Calculate mean reversion tendency (0-1)."""
        if len(prices) < 30:
            return 0.5
        
        # Hurst exponent estimation (simplified)
        returns = np.diff(prices) / prices[:-1]
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 20:
            return 0.5
        
        # Calculate autocorrelation at lag 1
        mean_return = np.mean(returns)
        var_return = np.var(returns)
        
        if var_return == 0:
            return 0.5
        
        autocorr = np.sum((returns[:-1] - mean_return) * (returns[1:] - mean_return)) / \
                   ((len(returns) - 1) * var_return)
        
        # Negative autocorrelation suggests mean reversion
        mean_rev_score = 0.5 - autocorr / 2
        
        return max(0.0, min(1.0, mean_rev_score))
    
    def _classify_regime(
        self,
        volatility: float,
        trend: float,
        mean_rev: float
    ) -> Tuple[MarketRegime, float]:
        """Classify regime based on scores."""
        
        # High volatility override
        if volatility > 0.8:
            return MarketRegime.HIGH_VOLATILITY, 0.8
        
        # Low volatility
        if volatility < 0.2:
            return MarketRegime.LOW_VOLATILITY, 0.7
        
        # Strong trend
        if trend > 0.4:
            return MarketRegime.TRENDING_UP, 0.7 + trend * 0.2
        if trend < -0.4:
            return MarketRegime.TRENDING_DOWN, 0.7 + abs(trend) * 0.2
        
        # Mean reverting
        if mean_rev > 0.7:
            return MarketRegime.MEAN_REVERTING, 0.6 + mean_rev * 0.2
        
        # Default to ranging
        return MarketRegime.RANGING, 0.5
    
    def get_strategy_params(
        self,
        market_id: str
    ) -> Dict[str, Any]:
        """Get recommended strategy parameters for current regime."""
        state = self._market_regimes.get(market_id)
        
        if not state:
            return self.REGIME_STRATEGIES[MarketRegime.RANGING]
        
        params = self.REGIME_STRATEGIES.get(
            state.primary,
            self.REGIME_STRATEGIES[MarketRegime.RANGING]
        ).copy()
        
        # Adjust by confidence
        params["position_size"] *= state.confidence
        params["regime"] = state.primary.value
        params["confidence"] = state.confidence
        
        return params
    
    def get_regime(self, market_id: str) -> Optional[RegimeState]:
        """Get current regime for market."""
        return self._market_regimes.get(market_id)
    
    def detect_regime_change(
        self,
        market_id: str,
        lookback: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Detect if regime has changed recently."""
        relevant = [
            s for s in self._regime_history
            if s.metadata.get("market_id") == market_id
        ][-lookback:]
        
        if len(relevant) < 3:
            return None
        
        current = relevant[-1].primary
        previous = [s.primary for s in relevant[:-1]]
        
        # Check if current regime is different from most of previous
        if previous.count(current) < len(previous) / 2:
            return {
                "changed": True,
                "from": previous[-1].value,
                "to": current.value,
                "timestamp": relevant[-1].timestamp.isoformat()
            }
        
        return None
    
    def get_all_regimes(self) -> Dict[str, Dict[str, Any]]:
        """Get current regimes for all markets."""
        return {
            market_id: state.to_dict()
            for market_id, state in self._market_regimes.items()
        }

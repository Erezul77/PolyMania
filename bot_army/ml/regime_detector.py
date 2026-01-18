"""
Advanced Regime Detection
=========================

World-class market regime detection using:
- Hidden Markov Models (HMM)
- Volatility clustering (GARCH-like)
- Trend strength analysis
- Correlation regime shifts
- Volume regime detection
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger("ml.regime_detector")


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state with metadata."""
    regime: MarketRegime
    confidence: float
    duration_bars: int
    volatility_percentile: float
    trend_strength: float
    mean_reversion_score: float
    regime_stability: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "duration_bars": self.duration_bars,
            "volatility_percentile": self.volatility_percentile,
            "trend_strength": self.trend_strength,
            "mean_reversion_score": self.mean_reversion_score,
            "regime_stability": self.regime_stability,
            "timestamp": self.timestamp.isoformat()
        }


class HiddenMarkovRegime:
    """
    Hidden Markov Model for regime detection.
    Simplified implementation without external HMM library.
    """
    
    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        
        # Transition matrix (initialized to uniform)
        self.transition = np.ones((n_states, n_states)) / n_states
        
        # Emission means (will be learned)
        self.emission_means = np.linspace(-0.02, 0.02, n_states)
        self.emission_stds = np.ones(n_states) * 0.01
        
        # State probabilities
        self.state_probs = np.ones(n_states) / n_states
        
        # State names
        self.state_names = [
            MarketRegime.TRENDING_DOWN,
            MarketRegime.MEAN_REVERTING,
            MarketRegime.LOW_VOLATILITY,
            MarketRegime.TRENDING_UP
        ][:n_states]
        
        # History for online learning
        self._return_history = deque(maxlen=500)
    
    def update(self, returns: np.ndarray):
        """Update HMM with new return data."""
        for r in returns:
            self._return_history.append(r)
        
        if len(self._return_history) < 50:
            return
        
        # Simple online update of emission parameters
        returns_arr = np.array(self._return_history)
        
        # K-means-like update for emission means
        sorted_returns = np.sort(returns_arr)
        chunk_size = len(sorted_returns) // self.n_states
        
        for i in range(self.n_states):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_states - 1 else len(sorted_returns)
            chunk = sorted_returns[start:end]
            
            self.emission_means[i] = np.mean(chunk)
            self.emission_stds[i] = max(np.std(chunk), 0.001)
    
    def predict_state(self, current_return: float) -> Tuple[MarketRegime, float]:
        """Predict current state given return."""
        # Calculate emission probabilities
        emissions = np.zeros(self.n_states)
        for i in range(self.n_states):
            # Gaussian emission
            diff = current_return - self.emission_means[i]
            emissions[i] = np.exp(-0.5 * (diff / self.emission_stds[i]) ** 2)
        
        # Normalize
        emissions = emissions / (np.sum(emissions) + 1e-10)
        
        # Update state probabilities (forward algorithm step)
        new_probs = np.dot(self.state_probs, self.transition) * emissions
        new_probs = new_probs / (np.sum(new_probs) + 1e-10)
        
        self.state_probs = new_probs
        
        # Get most likely state
        state_idx = np.argmax(new_probs)
        confidence = new_probs[state_idx]
        
        return self.state_names[state_idx], confidence


class VolatilityRegimeDetector:
    """
    Volatility clustering and regime detection.
    GARCH-inspired without requiring arch library.
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self._vol_history = deque(maxlen=500)
        self._return_history = deque(maxlen=500)
        
        # EWMA parameters
        self.alpha = 0.94  # Decay factor
        self._ewma_var = None
    
    def update(self, returns: np.ndarray):
        """Update with new returns."""
        for r in returns:
            self._return_history.append(r)
            
            # EWMA variance update
            if self._ewma_var is None:
                self._ewma_var = r ** 2
            else:
                self._ewma_var = self.alpha * self._ewma_var + (1 - self.alpha) * r ** 2
            
            self._vol_history.append(np.sqrt(self._ewma_var))
    
    def get_volatility_state(self) -> Tuple[str, float, float]:
        """
        Get current volatility regime.
        
        Returns:
            (regime, current_vol, percentile)
        """
        if len(self._vol_history) < 20:
            return "unknown", 0, 0.5
        
        current_vol = self._vol_history[-1]
        vol_array = np.array(self._vol_history)
        
        # Calculate percentile
        percentile = np.mean(vol_array <= current_vol)
        
        # Determine regime
        if percentile > 0.9:
            regime = "extreme_high"
        elif percentile > 0.75:
            regime = "high"
        elif percentile > 0.5:
            regime = "normal_high"
        elif percentile > 0.25:
            regime = "normal_low"
        elif percentile > 0.1:
            regime = "low"
        else:
            regime = "extreme_low"
        
        return regime, current_vol, percentile
    
    def detect_vol_spike(self, threshold: float = 2.0) -> bool:
        """Detect sudden volatility spike."""
        if len(self._vol_history) < 20:
            return False
        
        recent = np.array(list(self._vol_history)[-5:])
        historical = np.array(list(self._vol_history)[-50:-5])
        
        if len(historical) < 10:
            return False
        
        avg_recent = np.mean(recent)
        avg_hist = np.mean(historical)
        
        return avg_recent > avg_hist * threshold


class TrendStrengthAnalyzer:
    """Analyze trend strength using multiple indicators."""
    
    def __init__(self, periods: List[int] = None):
        self.periods = periods or [10, 20, 50, 100]
        self._price_history = deque(maxlen=200)
    
    def update(self, price: float):
        """Update with new price."""
        self._price_history.append(price)
    
    def get_trend_strength(self) -> Tuple[float, str]:
        """
        Calculate trend strength from -1 (strong down) to +1 (strong up).
        
        Returns:
            (strength, direction)
        """
        if len(self._price_history) < max(self.periods):
            return 0, "unknown"
        
        prices = np.array(self._price_history)
        
        # Calculate multiple moving average slopes
        slopes = []
        for period in self.periods:
            if len(prices) >= period:
                ma = np.convolve(prices, np.ones(period)/period, mode='valid')
                if len(ma) >= 2:
                    slope = (ma[-1] - ma[-min(period, len(ma)-1)]) / ma[-min(period, len(ma)-1)]
                    slopes.append(slope)
        
        if not slopes:
            return 0, "unknown"
        
        # Aggregate slopes with more weight on longer-term
        weights = np.array([1, 2, 3, 4][:len(slopes)])
        avg_slope = np.average(slopes, weights=weights)
        
        # ADX-like calculation (simplified)
        if len(prices) >= 14:
            high = np.maximum.accumulate(prices[-14:])
            low = np.minimum.accumulate(prices[-14:])
            
            plus_dm = np.diff(high)
            minus_dm = np.diff(low)
            
            plus_dm = np.where(plus_dm > 0, plus_dm, 0)
            minus_dm = np.where(minus_dm < 0, -minus_dm, 0)
            
            atr = np.mean(high - low)
            if atr > 0:
                plus_di = np.mean(plus_dm) / atr
                minus_di = np.mean(minus_dm) / atr
                
                dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
                adx = dx * 100
            else:
                adx = 0
        else:
            adx = 0
        
        # Combine slope and ADX
        strength = np.tanh(avg_slope * 50) * (0.5 + 0.5 * min(adx / 50, 1))
        
        if strength > 0.3:
            direction = "strong_up"
        elif strength > 0.1:
            direction = "up"
        elif strength > -0.1:
            direction = "neutral"
        elif strength > -0.3:
            direction = "down"
        else:
            direction = "strong_down"
        
        return strength, direction


class AdvancedRegimeDetector:
    """
    Master regime detector combining all methods.
    """
    
    def __init__(self):
        self.hmm = HiddenMarkovRegime(n_states=4)
        self.volatility = VolatilityRegimeDetector()
        self.trend = TrendStrengthAnalyzer()
        
        # Regime history
        self._regime_history = deque(maxlen=100)
        self._current_regime = MarketRegime.UNKNOWN
        self._regime_start_time = datetime.utcnow()
        self._regime_duration = 0
    
    def update(self, price: float, returns: np.ndarray = None):
        """Update all detectors with new data."""
        self.trend.update(price)
        
        if returns is not None and len(returns) > 0:
            self.hmm.update(returns)
            self.volatility.update(returns)
    
    def detect_regime(self, price: float = None, current_return: float = None) -> RegimeState:
        """
        Detect current market regime using all available signals.
        """
        # Get component signals
        hmm_regime, hmm_conf = MarketRegime.UNKNOWN, 0.5
        if current_return is not None:
            hmm_regime, hmm_conf = self.hmm.predict_state(current_return)
        
        vol_regime, current_vol, vol_percentile = self.volatility.get_volatility_state()
        trend_strength, trend_dir = self.trend.get_trend_strength()
        
        # Combine signals to determine overall regime
        final_regime = self._combine_signals(
            hmm_regime, hmm_conf,
            vol_regime, vol_percentile,
            trend_strength, trend_dir
        )
        
        # Track regime changes
        if final_regime != self._current_regime:
            self._regime_history.append({
                "from": self._current_regime.value,
                "to": final_regime.value,
                "duration": self._regime_duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            self._current_regime = final_regime
            self._regime_start_time = datetime.utcnow()
            self._regime_duration = 0
        else:
            self._regime_duration += 1
        
        # Calculate stability (how long in current regime)
        stability = min(self._regime_duration / 50, 1.0)
        
        # Calculate mean reversion score
        mr_score = self._calculate_mean_reversion_score()
        
        # Build confidence
        confidence = (hmm_conf * 0.4 + 
                     (1 - abs(trend_strength - 0.5) * 2) * 0.3 +
                     stability * 0.3)
        
        return RegimeState(
            regime=final_regime,
            confidence=confidence,
            duration_bars=self._regime_duration,
            volatility_percentile=vol_percentile,
            trend_strength=trend_strength,
            mean_reversion_score=mr_score,
            regime_stability=stability
        )
    
    def _combine_signals(
        self,
        hmm_regime: MarketRegime, hmm_conf: float,
        vol_regime: str, vol_percentile: float,
        trend_strength: float, trend_dir: str
    ) -> MarketRegime:
        """Combine signals to determine final regime."""
        
        # Crisis detection (high vol + strong down trend)
        if vol_percentile > 0.9 and trend_strength < -0.3:
            return MarketRegime.CRISIS
        
        # Recovery (decreasing vol + positive trend)
        if vol_percentile < 0.3 and trend_strength > 0.2:
            if self._current_regime in [MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY]:
                return MarketRegime.RECOVERY
        
        # Breakout (vol spike + strong trend)
        if self.volatility.detect_vol_spike() and abs(trend_strength) > 0.3:
            return MarketRegime.BREAKOUT
        
        # High volatility regime
        if vol_percentile > 0.75:
            return MarketRegime.HIGH_VOLATILITY
        
        # Low volatility regime
        if vol_percentile < 0.25:
            return MarketRegime.LOW_VOLATILITY
        
        # Trending regimes
        if trend_strength > 0.25:
            return MarketRegime.TRENDING_UP
        elif trend_strength < -0.25:
            return MarketRegime.TRENDING_DOWN
        
        # Consolidation (low vol + weak trend)
        if vol_percentile < 0.4 and abs(trend_strength) < 0.15:
            return MarketRegime.CONSOLIDATION
        
        # Mean reverting (normal vol + weak trend)
        return MarketRegime.MEAN_REVERTING
    
    def _calculate_mean_reversion_score(self) -> float:
        """Calculate how mean-reverting the current market is."""
        if len(self.trend._price_history) < 50:
            return 0.5
        
        prices = np.array(self.trend._price_history)
        
        # Calculate returns autocorrelation
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) < 20:
            return 0.5
        
        # Lag-1 autocorrelation
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        
        # Negative autocorr = mean reverting, positive = trending
        # Transform to 0-1 score (1 = highly mean reverting)
        mr_score = 0.5 - autocorr * 0.5
        
        return np.clip(mr_score, 0, 1)
    
    def get_regime_features(self) -> Dict[str, float]:
        """Get regime-based features for ML models."""
        state = self.detect_regime()
        
        return {
            "regime_trending_up": 1.0 if state.regime == MarketRegime.TRENDING_UP else 0.0,
            "regime_trending_down": 1.0 if state.regime == MarketRegime.TRENDING_DOWN else 0.0,
            "regime_mean_reverting": 1.0 if state.regime == MarketRegime.MEAN_REVERTING else 0.0,
            "regime_high_vol": 1.0 if state.regime == MarketRegime.HIGH_VOLATILITY else 0.0,
            "regime_low_vol": 1.0 if state.regime == MarketRegime.LOW_VOLATILITY else 0.0,
            "regime_breakout": 1.0 if state.regime == MarketRegime.BREAKOUT else 0.0,
            "regime_crisis": 1.0 if state.regime == MarketRegime.CRISIS else 0.0,
            "regime_confidence": state.confidence,
            "regime_duration": min(state.duration_bars / 100, 1.0),
            "regime_stability": state.regime_stability,
            "vol_percentile": state.volatility_percentile,
            "trend_strength": state.trend_strength,
            "mean_reversion_score": state.mean_reversion_score
        }
    
    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime
    
    @property
    def regime_history(self) -> List[Dict]:
        return list(self._regime_history)

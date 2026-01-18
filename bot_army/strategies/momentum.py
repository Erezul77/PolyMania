"""Momentum trading strategy."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional

from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import Signal, SignalType

logger = logging.getLogger("strategy.momentum")


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.
    
    Identifies and trades in the direction of strong price movements.
    Uses multiple momentum indicators and volume confirmation.
    """
    
    DEFAULT_PARAMS = {
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "momentum_lookback": 10,
        "volume_multiplier": 1.5,
        "trend_strength_threshold": 0.1,
        "breakout_threshold": 0.02
    }
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="momentum",
                min_confidence=0.35,
                position_size_pct=0.08,
                stop_loss_pct=0.08,
                take_profit_pct=0.15
            )
        
        # Merge default params
        config.params = {**self.DEFAULT_PARAMS, **config.params}
        super().__init__(config)
    
    def analyze(
        self,
        market_id: str,
        features: Dict[str, float],
        price_history: List[float],
        orderbook: Dict = None,
        trades: List[Dict] = None
    ) -> Optional[StrategyResult]:
        """Analyze market for momentum signals."""
        
        if not self.should_generate_signal(market_id):
            return None
        
        if len(price_history) < 20:
            return None
        
        prices = np.array(price_history)
        current_price = prices[-1]
        
        # Calculate momentum indicators
        momentum_score = self._calculate_momentum(prices)
        trend_score = self._calculate_trend_strength(prices)
        rsi = features.get("rsi", 50)
        
        # Volume analysis
        volume_signal = 0
        if trades:
            volume_signal = self._analyze_volume(trades)
        
        # Orderbook analysis
        ob_signal = 0
        if orderbook:
            ob_signal = self._analyze_orderbook(orderbook)
        
        # Combine signals
        total_score = (
            momentum_score * 0.35 +
            trend_score * 0.25 +
            self._rsi_signal(rsi) * 0.2 +
            volume_signal * 0.1 +
            ob_signal * 0.1
        )
        
        # Determine signal
        reasoning = []
        signal_type = SignalType.HOLD
        confidence = abs(total_score)
        
        if total_score > self.config.params["trend_strength_threshold"]:
            signal_type = SignalType.BUY
            reasoning.append(f"Strong bullish momentum: {momentum_score:.2f}")
            if trend_score > 0:
                reasoning.append(f"Uptrend confirmed: {trend_score:.2f}")
            if rsi < 70:
                reasoning.append(f"RSI not overbought: {rsi:.0f}")
            if volume_signal > 0:
                reasoning.append("Volume confirming move")
        
        elif total_score < -self.config.params["trend_strength_threshold"]:
            signal_type = SignalType.SELL
            reasoning.append(f"Strong bearish momentum: {momentum_score:.2f}")
            if trend_score < 0:
                reasoning.append(f"Downtrend confirmed: {trend_score:.2f}")
            if rsi > 30:
                reasoning.append(f"RSI not oversold: {rsi:.0f}")
            if volume_signal < 0:
                reasoning.append("Volume confirming move")
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Create signal
        signal = self.create_signal(
            market_id=market_id,
            signal_type=signal_type,
            confidence=min(0.95, confidence),
            price=current_price,
            features={
                "momentum": momentum_score,
                "trend": trend_score,
                "rsi": rsi,
                "volume_signal": volume_signal,
                "ob_signal": ob_signal
            }
        )
        
        if not self.validate_signal(signal):
            return None
        
        return StrategyResult(
            signal=signal,
            strategy_name=self.config.name,
            score=total_score,
            reasoning=reasoning,
            metadata={
                "momentum_score": momentum_score,
                "trend_score": trend_score,
                "volume_signal": volume_signal
            }
        )
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate momentum score (-1 to 1)."""
        lookback = self.config.params["momentum_lookback"]
        
        if len(prices) < lookback:
            return 0.0
        
        # Rate of change
        roc = (prices[-1] - prices[-lookback]) / prices[-lookback]
        
        # Acceleration (2nd derivative)
        if len(prices) >= lookback * 2:
            prev_roc = (prices[-lookback] - prices[-lookback*2]) / prices[-lookback*2]
            acceleration = roc - prev_roc
        else:
            acceleration = 0
        
        # Combine
        momentum = roc * 0.7 + acceleration * 0.3
        
        # Normalize to -1 to 1
        return max(-1, min(1, momentum * 10))
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength (-1 to 1)."""
        if len(prices) < 20:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(prices[-20:]))
        slope, _ = np.polyfit(x, prices[-20:], 1)
        
        # Normalize by price level
        normalized_slope = slope / np.mean(prices[-20:])
        
        # ADX-like directional movement
        up_moves = sum(1 for i in range(1, min(20, len(prices))) 
                      if prices[-i] > prices[-i-1])
        directional = (up_moves - (20 - up_moves)) / 20
        
        # Combine
        trend = normalized_slope * 100 * 0.6 + directional * 0.4
        
        return max(-1, min(1, trend))
    
    def _rsi_signal(self, rsi: float) -> float:
        """Convert RSI to signal (-1 to 1)."""
        overbought = self.config.params["rsi_overbought"]
        oversold = self.config.params["rsi_oversold"]
        
        if rsi >= overbought:
            # Overbought - potential reversal (contrarian) or momentum continuation
            return -0.5 + (rsi - overbought) / 30 * -0.5  # More extreme = stronger
        elif rsi <= oversold:
            # Oversold - potential reversal (contrarian) or momentum continuation
            return 0.5 + (oversold - rsi) / 30 * 0.5
        else:
            # Neutral zone - slight bias based on direction
            return (rsi - 50) / 50 * 0.3
    
    def _analyze_volume(self, trades: List[Dict]) -> float:
        """Analyze volume for confirmation (-1 to 1)."""
        if not trades:
            return 0.0
        
        buy_volume = sum(t.get("value", 0) for t in trades 
                        if t.get("side", "").upper() == "BUY")
        sell_volume = sum(t.get("value", 0) for t in trades 
                         if t.get("side", "").upper() == "SELL")
        
        total = buy_volume + sell_volume
        if total == 0:
            return 0.0
        
        # Net flow direction
        net_flow = (buy_volume - sell_volume) / total
        
        # Volume relative to expected
        multiplier = self.config.params["volume_multiplier"]
        # Assume trades have 'expected_volume' or calculate from history
        
        return net_flow
    
    def _analyze_orderbook(self, orderbook: Dict) -> float:
        """Analyze orderbook for momentum confirmation (-1 to 1)."""
        imbalance = orderbook.get("imbalance", 0)
        
        # Strong imbalance suggests momentum direction
        return imbalance * 0.5


class AdaptiveMomentumStrategy(MomentumStrategy):
    """
    Adaptive momentum strategy that adjusts to market conditions.
    """
    
    def __init__(self, config: StrategyConfig = None):
        super().__init__(config)
        self._regime = "normal"
        self._adaptive_params = {}
    
    def set_regime(self, regime: str, volatility: float):
        """Adapt strategy to market regime."""
        self._regime = regime
        
        if regime == "HIGH_VOLATILITY":
            self._adaptive_params = {
                "trend_strength_threshold": 0.4,  # Higher threshold
                "position_size_pct": 0.05,  # Smaller positions
                "stop_loss_pct": 0.12  # Wider stops
            }
        elif regime == "LOW_VOLATILITY":
            self._adaptive_params = {
                "trend_strength_threshold": 0.2,  # Lower threshold
                "position_size_pct": 0.12,  # Larger positions
                "stop_loss_pct": 0.05  # Tighter stops
            }
        else:
            self._adaptive_params = {}
        
        # Apply adaptations
        for key, value in self._adaptive_params.items():
            if key in self.config.params:
                self.config.params[key] = value
            elif hasattr(self.config, key):
                setattr(self.config, key, value)

"""Breakout Strategy - Detects price breakouts from ranges."""

import numpy as np
from typing import Any, Dict, List, Optional
from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import SignalType


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy that identifies and trades range breakouts.
    
    Signals:
    - BUY: Price breaks above resistance with volume confirmation
    - SELL: Price breaks below support with volume confirmation
    """
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="breakout",
                min_confidence=0.55,
                position_size_pct=0.08,
                stop_loss_pct=0.08,
                take_profit_pct=0.15,
                cooldown_seconds=600
            )
        super().__init__(config)
        
        # Parameters
        self.lookback = config.params.get("lookback", 20)
        self.breakout_threshold = config.params.get("breakout_threshold", 0.02)
        self.volume_multiplier = config.params.get("volume_multiplier", 1.5)
    
    def analyze(
        self,
        market_id: str,
        features: Dict[str, float],
        price_history: List[float],
        orderbook: Dict = None,
        trades: List[Dict] = None
    ) -> Optional[StrategyResult]:
        
        if not self.should_generate_signal(market_id):
            return None
        
        if len(price_history) < self.lookback + 5:
            return None
        
        # Calculate range
        lookback_prices = price_history[-self.lookback:-1]
        current_price = price_history[-1]
        
        high = max(lookback_prices)
        low = min(lookback_prices)
        range_size = high - low
        
        if range_size < 0.01:  # Too tight range
            return None
        
        # Check for breakout
        breakout_up = current_price > high * (1 + self.breakout_threshold * 0.5)
        breakout_down = current_price < low * (1 - self.breakout_threshold * 0.5)
        
        if not breakout_up and not breakout_down:
            return None
        
        # Volume confirmation
        volume_confirmed = True
        if features.get("trade_total_volume", 0) > 0:
            avg_volume = features.get("trade_total_volume", 100) / max(1, features.get("trade_count", 1))
            volume_confirmed = avg_volume > self.volume_multiplier
        
        # Calculate confidence
        breakout_strength = abs(current_price - (high if breakout_up else low)) / range_size
        confidence = min(0.9, 0.5 + breakout_strength * 0.3 + (0.1 if volume_confirmed else 0))
        
        signal_type = SignalType.BUY if breakout_up else SignalType.SELL
        
        signal = self.create_signal(
            market_id=market_id,
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            features=features
        )
        
        return StrategyResult(
            signal=signal,
            strategy_name=self.config.name,
            score=confidence,
            reasoning=[
                f"{'Upside' if breakout_up else 'Downside'} breakout detected",
                f"Range: {low:.4f} - {high:.4f}",
                f"Breakout strength: {breakout_strength:.2%}",
                f"Volume confirmed: {volume_confirmed}"
            ]
        )

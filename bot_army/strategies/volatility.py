"""Volatility Strategy - Trades volatility expansion/contraction."""

import numpy as np
from typing import Any, Dict, List, Optional
from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import SignalType


class VolatilityStrategy(BaseStrategy):
    """
    Volatility-based strategy using Bollinger Band squeeze and expansion.
    
    Signals:
    - BUY: Volatility squeeze followed by upside expansion
    - SELL: Volatility squeeze followed by downside expansion
    """
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="volatility",
                min_confidence=0.5,
                position_size_pct=0.08,
                stop_loss_pct=0.1,
                take_profit_pct=0.2,
                cooldown_seconds=600
            )
        super().__init__(config)
        
        self.bb_period = config.params.get("bb_period", 20)
        self.bb_std = config.params.get("bb_std", 2.0)
        self.squeeze_threshold = config.params.get("squeeze_threshold", 0.1)
    
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
        
        if len(price_history) < self.bb_period + 10:
            return None
        
        prices = np.array(price_history)
        current_price = prices[-1]
        
        # Current Bollinger Bands
        bb_prices = prices[-self.bb_period:]
        bb_mean = np.mean(bb_prices)
        bb_std = np.std(bb_prices)
        bb_upper = bb_mean + self.bb_std * bb_std
        bb_lower = bb_mean - self.bb_std * bb_std
        bb_width = (bb_upper - bb_lower) / bb_mean
        
        # Previous BB width (5 periods ago)
        prev_prices = prices[-(self.bb_period + 5):-5]
        prev_mean = np.mean(prev_prices)
        prev_std = np.std(prev_prices)
        prev_width = (2 * self.bb_std * prev_std) / prev_mean
        
        # Detect squeeze release
        was_squeezed = prev_width < self.squeeze_threshold
        is_expanding = bb_width > prev_width * 1.3
        
        if not (was_squeezed and is_expanding):
            # Also check for band touch breakout
            touch_upper = current_price >= bb_upper * 0.98
            touch_lower = current_price <= bb_lower * 1.02
            if not touch_upper and not touch_lower:
                return None
        
        # Direction
        momentum = (current_price - prices[-5]) / prices[-5]
        
        if momentum > 0.01:
            signal_type = SignalType.BUY
        elif momentum < -0.01:
            signal_type = SignalType.SELL
        else:
            return None
        
        # Confidence
        expansion_factor = bb_width / max(prev_width, 0.01)
        momentum_strength = min(1, abs(momentum) * 20)
        confidence = min(0.85, 0.45 + expansion_factor * 0.1 + momentum_strength * 0.2)
        
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
                f"Volatility {'expansion' if is_expanding else 'band touch'}",
                f"BB width: {bb_width:.3f} (was {prev_width:.3f})",
                f"Momentum: {momentum:.2%}",
                f"Expansion factor: {expansion_factor:.2f}x"
            ]
        )

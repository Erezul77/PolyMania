"""Trend Following Strategy - Follows established trends."""

import numpy as np
from typing import Any, Dict, List, Optional
from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import SignalType


class TrendFollowingStrategy(BaseStrategy):
    """
    Classic trend following using multiple moving averages.
    
    Signals:
    - BUY: Strong uptrend with MA alignment and momentum
    - SELL: Strong downtrend with MA alignment and momentum
    """
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="trend_following",
                min_confidence=0.5,
                position_size_pct=0.1,
                stop_loss_pct=0.12,
                take_profit_pct=0.25,
                cooldown_seconds=900
            )
        super().__init__(config)
        
        self.fast_period = config.params.get("fast_period", 5)
        self.medium_period = config.params.get("medium_period", 10)
        self.slow_period = config.params.get("slow_period", 20)
        self.adx_threshold = config.params.get("adx_threshold", 25)
    
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
        
        if len(price_history) < self.slow_period + 5:
            return None
        
        prices = np.array(price_history)
        current_price = prices[-1]
        
        # Calculate MAs
        fast_ma = np.mean(prices[-self.fast_period:])
        medium_ma = np.mean(prices[-self.medium_period:])
        slow_ma = np.mean(prices[-self.slow_period:])
        
        # Check MA alignment
        bullish_alignment = fast_ma > medium_ma > slow_ma
        bearish_alignment = fast_ma < medium_ma < slow_ma
        
        if not bullish_alignment and not bearish_alignment:
            return None
        
        # Trend strength (pseudo-ADX)
        returns = np.diff(prices[-self.slow_period:])
        trend_strength = abs(np.mean(returns)) / (np.std(returns) + 1e-8) * 100
        
        if trend_strength < self.adx_threshold * 0.5:
            return None
        
        # Price position relative to MAs
        ma_spread = abs(fast_ma - slow_ma) / slow_ma
        price_above_all = current_price > fast_ma > medium_ma > slow_ma
        price_below_all = current_price < fast_ma < medium_ma < slow_ma
        
        if bullish_alignment and not price_above_all:
            return None
        if bearish_alignment and not price_below_all:
            return None
        
        # Confidence calculation
        confidence = min(0.85, 0.45 + ma_spread * 5 + trend_strength / 100)
        
        signal_type = SignalType.BUY if bullish_alignment else SignalType.SELL
        
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
                f"{'Bullish' if bullish_alignment else 'Bearish'} trend alignment",
                f"MA spread: {ma_spread:.2%}",
                f"Trend strength: {trend_strength:.1f}",
                f"Fast/Med/Slow MA: {fast_ma:.4f}/{medium_ma:.4f}/{slow_ma:.4f}"
            ]
        )

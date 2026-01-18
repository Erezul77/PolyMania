"""Counter-Trend Strategy - Fades extreme moves."""

import numpy as np
from typing import Any, Dict, List, Optional
from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import SignalType


class CounterTrendStrategy(BaseStrategy):
    """
    Counter-trend strategy that fades extreme price moves.
    
    Signals:
    - BUY: Extreme oversold conditions, expecting bounce
    - SELL: Extreme overbought conditions, expecting pullback
    """
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="counter_trend",
                min_confidence=0.55,
                position_size_pct=0.06,
                stop_loss_pct=0.08,
                take_profit_pct=0.12,
                cooldown_seconds=600
            )
        super().__init__(config)
        
        self.rsi_oversold = config.params.get("rsi_oversold", 25)
        self.rsi_overbought = config.params.get("rsi_overbought", 75)
        self.zscore_threshold = config.params.get("zscore_threshold", 2.0)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-(period + 1):])
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
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
        
        if len(price_history) < 25:
            return None
        
        prices = np.array(price_history)
        current_price = prices[-1]
        
        # RSI
        rsi = self._calculate_rsi(prices)
        
        # Z-score
        mean_price = np.mean(prices[-20:])
        std_price = np.std(prices[-20:])
        zscore = (current_price - mean_price) / (std_price + 1e-8)
        
        # Check for extreme conditions
        oversold = rsi < self.rsi_oversold and zscore < -self.zscore_threshold * 0.8
        overbought = rsi > self.rsi_overbought and zscore > self.zscore_threshold * 0.8
        
        if not oversold and not overbought:
            return None
        
        # Look for reversal candle (price moving back toward mean)
        recent_move = (current_price - prices[-3]) / prices[-3]
        reversal_forming = (oversold and recent_move > 0) or (overbought and recent_move < 0)
        
        # Confidence
        extreme_level = abs(zscore) / self.zscore_threshold
        rsi_extreme = abs(rsi - 50) / 50
        confidence = min(0.8, 0.4 + extreme_level * 0.15 + rsi_extreme * 0.15 + (0.1 if reversal_forming else 0))
        
        signal_type = SignalType.BUY if oversold else SignalType.SELL
        
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
                f"{'Oversold' if oversold else 'Overbought'} extreme",
                f"RSI: {rsi:.1f}",
                f"Z-score: {zscore:.2f}",
                f"Reversal forming: {reversal_forming}"
            ]
        )

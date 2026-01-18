"""Volume Profile Strategy - Volume-based signals."""

import numpy as np
from typing import Any, Dict, List, Optional
from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import SignalType


class VolumeProfileStrategy(BaseStrategy):
    """
    Volume-based strategy using order flow and volume analysis.
    
    Signals:
    - BUY: High buy volume, positive order flow imbalance
    - SELL: High sell volume, negative order flow imbalance
    """
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="volume_profile",
                min_confidence=0.5,
                position_size_pct=0.08,
                stop_loss_pct=0.08,
                take_profit_pct=0.15,
                cooldown_seconds=450
            )
        super().__init__(config)
        
        self.volume_spike_mult = config.params.get("volume_spike_mult", 2.0)
        self.imbalance_threshold = config.params.get("imbalance_threshold", 0.3)
    
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
        
        if len(price_history) < 10:
            return None
        
        current_price = price_history[-1]
        
        # Volume features
        buy_volume = features.get("trade_buy_volume", 0)
        sell_volume = features.get("trade_sell_volume", 0)
        total_volume = buy_volume + sell_volume
        
        if total_volume < 100:  # Minimum volume threshold
            return None
        
        # Volume imbalance
        imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        # Large trade detection
        large_buy = features.get("trade_large_buy", 0)
        large_sell = features.get("trade_large_sell", 0)
        large_flow = large_buy - large_sell
        
        # Orderbook imbalance
        ob_imbalance = features.get("ob_imbalance", 0)
        bid_ratio = features.get("ob_bid_ratio", 0.5)
        
        # Combine signals
        buy_pressure = (
            imbalance > self.imbalance_threshold or
            (large_flow > 0 and bid_ratio > 0.6)
        )
        sell_pressure = (
            imbalance < -self.imbalance_threshold or
            (large_flow < 0 and bid_ratio < 0.4)
        )
        
        if not buy_pressure and not sell_pressure:
            return None
        
        # Price confirmation
        price_move = (current_price - price_history[-5]) / price_history[-5]
        price_confirms = (buy_pressure and price_move > 0) or (sell_pressure and price_move < 0)
        
        # Confidence
        imbalance_strength = min(1, abs(imbalance) / 0.5)
        confidence = min(0.8, 0.4 + imbalance_strength * 0.2 + (0.15 if price_confirms else 0) + abs(ob_imbalance) * 0.1)
        
        signal_type = SignalType.BUY if buy_pressure else SignalType.SELL
        
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
                f"{'Buy' if buy_pressure else 'Sell'} pressure detected",
                f"Volume imbalance: {imbalance:.2%}",
                f"Large trade flow: ${large_flow:,.0f}",
                f"OB bid ratio: {bid_ratio:.1%}",
                f"Price confirms: {price_confirms}"
            ]
        )

"""Event-Driven Strategy - Trades around market events."""

import numpy as np
from typing import Any, Dict, List, Optional
from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import SignalType


class EventDrivenStrategy(BaseStrategy):
    """
    Event-driven strategy that trades around significant market events.
    
    Signals:
    - BUY: Event ending soon with positive momentum
    - SELL: Event ending soon with negative momentum or resolution expected
    """
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="event_driven",
                min_confidence=0.55,
                position_size_pct=0.06,
                stop_loss_pct=0.12,
                take_profit_pct=0.2,
                cooldown_seconds=1200
            )
        super().__init__(config)
        
        self.hours_before_end = config.params.get("hours_before_end", 48)
        self.volume_threshold = config.params.get("volume_threshold", 10000)
    
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
        
        # Event features
        hours_to_end = features.get("event_hours_to_end", float("inf"))
        event_volume = features.get("event_volume", 0)
        ending_soon = features.get("event_ending_soon", 0)
        ending_imminent = features.get("event_ending_imminent", 0)
        
        # Check if event-related trade opportunity
        is_event_trade = (
            hours_to_end < self.hours_before_end or
            ending_soon == 1 or
            event_volume > self.volume_threshold
        )
        
        if not is_event_trade:
            return None
        
        # Price position (for prediction markets, price = probability)
        is_extreme = current_price < 0.15 or current_price > 0.85
        is_uncertain = 0.35 < current_price < 0.65
        
        # Momentum into event
        momentum = (current_price - price_history[-10]) / price_history[-10] if len(price_history) >= 10 else 0
        strong_momentum = abs(momentum) > 0.05
        
        # Decision logic
        signal_type = None
        reasoning = []
        
        if ending_imminent and is_extreme:
            # Event ending very soon with extreme price - fade it (mean reversion)
            signal_type = SignalType.SELL if current_price > 0.85 else SignalType.BUY
            reasoning.append("Event imminent + extreme price = fade")
        elif ending_soon and strong_momentum:
            # Follow momentum into expiry
            signal_type = SignalType.BUY if momentum > 0 else SignalType.SELL
            reasoning.append("Follow momentum into expiry")
        elif event_volume > self.volume_threshold * 2 and is_uncertain:
            # High volume with uncertain outcome - follow volume direction
            vol_direction = features.get("trade_net_flow", 0)
            if vol_direction > 1000:
                signal_type = SignalType.BUY
            elif vol_direction < -1000:
                signal_type = SignalType.SELL
            reasoning.append("High volume event with uncertain outcome")
        
        if signal_type is None:
            return None
        
        # Confidence based on time to expiry and conviction
        time_factor = max(0.1, 1 - hours_to_end / self.hours_before_end)
        price_conviction = abs(current_price - 0.5) * 2
        confidence = min(0.8, 0.4 + time_factor * 0.2 + price_conviction * 0.15)
        
        signal = self.create_signal(
            market_id=market_id,
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            features=features
        )
        
        reasoning.extend([
            f"Hours to end: {hours_to_end:.1f}",
            f"Event volume: ${event_volume:,.0f}",
            f"Price: {current_price:.2%} (conviction: {price_conviction:.1%})"
        ])
        
        return StrategyResult(
            signal=signal,
            strategy_name=self.config.name,
            score=confidence,
            reasoning=reasoning
        )

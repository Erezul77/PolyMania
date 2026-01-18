"""Mean Reversion Strategy."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional

from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import Signal, SignalType

logger = logging.getLogger("strategy.mean_reversion")


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy.
    
    Trades against extreme moves expecting prices to return to average.
    Works best in ranging/mean-reverting market regimes.
    """
    
    DEFAULT_PARAMS = {
        "bb_threshold": 0.7,  # Trade when outside this BB percentile
        "rsi_upper": 65,
        "rsi_lower": 35,
        "mean_window": 20,
        "entry_deviation": 0.03,  # 3% from mean
        "profit_target_pct": 0.05,  # Target 5% move back
        "max_holding_periods": 20
    }
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="mean_reversion",
                min_confidence=0.40,
                position_size_pct=0.08,
                stop_loss_pct=0.12,  # Wider stops for mean reversion
                take_profit_pct=0.06
            )
        
        config.params = {**self.DEFAULT_PARAMS, **config.params}
        super().__init__(config)
        
        self._holding_periods: Dict[str, int] = {}
    
    def analyze(
        self,
        market_id: str,
        features: Dict[str, float],
        price_history: List[float],
        orderbook: Dict = None,
        trades: List[Dict] = None
    ) -> Optional[StrategyResult]:
        """Analyze market for mean reversion opportunities."""
        
        if not self.should_generate_signal(market_id):
            return None
        
        if len(price_history) < 30:
            return None
        
        prices = np.array(price_history)
        current_price = prices[-1]
        
        # Calculate mean and standard deviation
        window = self.config.params["mean_window"]
        mean = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        
        if std == 0:
            return None
        
        # Z-score
        zscore = (current_price - mean) / std
        
        # Technical indicators
        rsi = features.get("rsi", 50)
        bb_position = features.get("bb_position", 0.5)
        
        reasoning = []
        signal_type = SignalType.HOLD
        confidence = 0
        
        # Entry conditions
        bb_threshold = self.config.params["bb_threshold"]
        rsi_upper = self.config.params["rsi_upper"]
        rsi_lower = self.config.params["rsi_lower"]
        
        # Overbought - expect pullback
        if bb_position > bb_threshold and rsi > rsi_upper:
            signal_type = SignalType.SELL
            confidence = self._calculate_confidence(zscore, rsi, bb_position, "overbought")
            reasoning.append(f"Overbought: RSI={rsi:.0f}, BB={bb_position:.2f}")
            reasoning.append(f"Z-score: {zscore:.2f} - expecting mean reversion")
        
        # Oversold - expect bounce
        elif bb_position < (1 - bb_threshold) and rsi < rsi_lower:
            signal_type = SignalType.BUY
            confidence = self._calculate_confidence(zscore, rsi, bb_position, "oversold")
            reasoning.append(f"Oversold: RSI={rsi:.0f}, BB={bb_position:.2f}")
            reasoning.append(f"Z-score: {zscore:.2f} - expecting mean reversion")
        
        # Alternative: Price deviation from mean
        deviation = (current_price - mean) / mean
        entry_dev = self.config.params["entry_deviation"]
        
        if signal_type == SignalType.HOLD:
            if deviation > entry_dev:
                signal_type = SignalType.SELL
                confidence = min(0.7, abs(deviation) * 3)
                reasoning.append(f"Price {deviation:.1%} above mean")
            elif deviation < -entry_dev:
                signal_type = SignalType.BUY
                confidence = min(0.7, abs(deviation) * 3)
                reasoning.append(f"Price {abs(deviation):.1%} below mean")
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Create signal with mean as target
        signal = self.create_signal(
            market_id=market_id,
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            features={
                "zscore": zscore,
                "rsi": rsi,
                "bb_position": bb_position,
                "deviation": deviation
            }
        )
        
        # Override take profit to target the mean
        if signal_type == SignalType.SELL:
            signal.take_profit = mean * (1 - 0.02)  # Slightly below mean
        else:
            signal.take_profit = mean * (1 + 0.02)  # Slightly above mean
        
        signal.metadata["mean_target"] = mean
        
        if not self.validate_signal(signal):
            return None
        
        return StrategyResult(
            signal=signal,
            strategy_name=self.config.name,
            score=confidence,
            reasoning=reasoning,
            metadata={
                "zscore": zscore,
                "mean": mean,
                "std": std,
                "deviation": deviation
            }
        )
    
    def _calculate_confidence(
        self,
        zscore: float,
        rsi: float,
        bb_position: float,
        condition: str
    ) -> float:
        """Calculate confidence based on multiple factors."""
        
        # Base confidence from z-score
        zscore_conf = min(0.4, abs(zscore) / 5)
        
        # RSI contribution
        if condition == "overbought":
            rsi_conf = min(0.3, (rsi - 70) / 60)
        else:
            rsi_conf = min(0.3, (30 - rsi) / 60)
        
        # BB contribution
        if condition == "overbought":
            bb_conf = min(0.3, (bb_position - 0.8) / 0.4)
        else:
            bb_conf = min(0.3, (0.2 - bb_position) / 0.4)
        
        total = zscore_conf + rsi_conf + bb_conf
        return min(0.85, max(0.4, total))
    
    def should_exit(
        self,
        market_id: str,
        current_price: float,
        features: Dict[str, float]
    ) -> bool:
        """Check if position should be exited."""
        
        # Track holding periods
        if market_id not in self._holding_periods:
            self._holding_periods[market_id] = 0
        
        self._holding_periods[market_id] += 1
        
        # Exit if held too long
        if self._holding_periods[market_id] >= self.config.params["max_holding_periods"]:
            self._holding_periods[market_id] = 0
            return True
        
        # Exit if returned to mean zone
        bb_position = features.get("bb_position", 0.5)
        rsi = features.get("rsi", 50)
        
        if 0.3 < bb_position < 0.7 and 40 < rsi < 60:
            self._holding_periods[market_id] = 0
            return True
        
        return False

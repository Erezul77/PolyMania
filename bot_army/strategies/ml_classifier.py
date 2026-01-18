"""ML Classifier Strategy - Pure ML-based trading decisions."""

import numpy as np
from typing import Any, Dict, List, Optional
from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import SignalType


class MLClassifierStrategy(BaseStrategy):
    """
    Pure ML-driven strategy using trained models.
    
    Signals based on ML prediction confidence and direction.
    Requires ML features to be computed and passed in.
    """
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="ml_classifier",
                min_confidence=0.6,
                position_size_pct=0.08,
                stop_loss_pct=0.1,
                take_profit_pct=0.18,
                cooldown_seconds=600
            )
        super().__init__(config)
        
        self.prediction_threshold = config.params.get("prediction_threshold", 0.6)
        self.confidence_threshold = config.params.get("confidence_threshold", 0.15)
        self.require_feature_agreement = config.params.get("require_feature_agreement", True)
    
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
        
        if len(price_history) < 20:
            return None
        
        current_price = price_history[-1]
        
        # Get ML predictions
        ml_prediction = features.get("ml_prediction", 0.5)
        ml_confidence = features.get("ml_confidence", 0)
        
        # Check if ML has a strong opinion
        bullish_ml = ml_prediction > self.prediction_threshold and ml_confidence > self.confidence_threshold
        bearish_ml = ml_prediction < (1 - self.prediction_threshold) and ml_confidence > self.confidence_threshold
        
        if not bullish_ml and not bearish_ml:
            return None
        
        # Feature agreement check
        if self.require_feature_agreement:
            # Technical features should align
            rsi = features.get("rsi", 50)
            trend_strength = features.get("trend_strength", 0)
            momentum = features.get("momentum_5", 0)
            
            tech_bullish = rsi < 70 and trend_strength > -0.3 and momentum > -0.05
            tech_bearish = rsi > 30 and trend_strength < 0.3 and momentum < 0.05
            
            if bullish_ml and not tech_bullish:
                return None
            if bearish_ml and not tech_bearish:
                return None
        
        # Position in price range
        prices = np.array(price_history[-20:])
        price_percentile = (current_price - prices.min()) / (prices.max() - prices.min() + 1e-8)
        
        # Don't buy at the top, don't sell at the bottom
        if bullish_ml and price_percentile > 0.9:
            return None
        if bearish_ml and price_percentile < 0.1:
            return None
        
        # Confidence calculation
        base_conf = 0.4 + ml_confidence
        range_factor = 1 - abs(price_percentile - 0.5) * 0.3  # Better mid-range
        confidence = min(0.85, base_conf * range_factor)
        
        signal_type = SignalType.BUY if bullish_ml else SignalType.SELL
        
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
                f"ML prediction: {ml_prediction:.3f}",
                f"ML confidence: {ml_confidence:.3f}",
                f"Price percentile: {price_percentile:.1%}",
                f"Signal: {'BULLISH' if bullish_ml else 'BEARISH'}"
            ]
        )

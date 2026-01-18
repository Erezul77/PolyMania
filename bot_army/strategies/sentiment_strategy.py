"""Sentiment Strategy - Trades based on sentiment signals."""

import numpy as np
from typing import Any, Dict, List, Optional
from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import SignalType


class SentimentStrategy(BaseStrategy):
    """
    Sentiment-based strategy using external sentiment signals.
    
    Signals:
    - BUY: Strong positive sentiment with price confirmation
    - SELL: Strong negative sentiment with price confirmation
    """
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="sentiment",
                min_confidence=0.5,
                position_size_pct=0.07,
                stop_loss_pct=0.1,
                take_profit_pct=0.18,
                cooldown_seconds=900
            )
        super().__init__(config)
        
        self.sentiment_threshold = config.params.get("sentiment_threshold", 0.3)
        self.require_price_confirm = config.params.get("require_price_confirm", True)
    
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
        
        # Get sentiment features (from external signals or ML)
        sentiment_score = features.get("sentiment_score", 0)
        sentiment_magnitude = features.get("sentiment_magnitude", 0)
        news_sentiment = features.get("news_sentiment", 0)
        social_sentiment = features.get("social_sentiment", 0)
        
        # Aggregate sentiment
        if sentiment_score == 0 and news_sentiment == 0 and social_sentiment == 0:
            # Use ML prediction as proxy for sentiment
            ml_pred = features.get("ml_prediction", 0.5)
            ml_conf = features.get("ml_confidence", 0)
            sentiment_score = (ml_pred - 0.5) * 2 * ml_conf
        
        # Check sentiment strength
        bullish_sentiment = sentiment_score > self.sentiment_threshold
        bearish_sentiment = sentiment_score < -self.sentiment_threshold
        
        if not bullish_sentiment and not bearish_sentiment:
            return None
        
        # Price confirmation
        price_move = (current_price - price_history[-5]) / price_history[-5]
        price_confirms = (bullish_sentiment and price_move > 0) or (bearish_sentiment and price_move < 0)
        
        if self.require_price_confirm and not price_confirms:
            # Allow contrarian if sentiment is extreme
            if abs(sentiment_score) < 0.7:
                return None
        
        # Confidence
        sentiment_strength = min(1, abs(sentiment_score))
        confidence = min(0.8, 0.4 + sentiment_strength * 0.3 + (0.1 if price_confirms else 0))
        
        signal_type = SignalType.BUY if bullish_sentiment else SignalType.SELL
        
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
                f"{'Bullish' if bullish_sentiment else 'Bearish'} sentiment",
                f"Sentiment score: {sentiment_score:.2f}",
                f"Price confirms: {price_confirms}",
                f"Sentiment strength: {sentiment_strength:.1%}"
            ]
        )

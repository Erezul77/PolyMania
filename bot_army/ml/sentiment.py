"""
External Sentiment Integration
==============================

Integrates sentiment from multiple sources:
- Social media (Twitter/X mentions)
- News headlines
- Market fear/greed indicators
- On-chain metrics (for crypto)
- Options flow
- Analyst ratings
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import asyncio
import aiohttp
import re

logger = logging.getLogger("ml.sentiment")


class SentimentSource(Enum):
    """Sentiment data sources."""
    SOCIAL = "social"
    NEWS = "news"
    FEAR_GREED = "fear_greed"
    ONCHAIN = "onchain"
    OPTIONS = "options"
    ANALYST = "analyst"


@dataclass
class SentimentReading:
    """Single sentiment reading."""
    source: SentimentSource
    value: float  # -1 (extreme fear) to +1 (extreme greed)
    confidence: float  # 0-1
    sample_size: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source.value,
            "value": self.value,
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AggregateSentiment:
    """Aggregated sentiment from all sources."""
    overall: float  # -1 to +1
    confidence: float  # 0-1
    readings: Dict[str, SentimentReading]
    contrarian_signal: bool  # True if extreme sentiment suggests reversal
    trend: str  # "improving", "worsening", "stable"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "overall": self.overall,
            "confidence": self.confidence,
            "readings": {k: v.to_dict() for k, v in self.readings.items()},
            "contrarian_signal": self.contrarian_signal,
            "trend": self.trend,
            "timestamp": self.timestamp.isoformat()
        }


class TextSentimentAnalyzer:
    """
    Simple text sentiment analyzer.
    Uses lexicon-based approach without ML dependencies.
    """
    
    def __init__(self):
        # Positive/negative word lists
        self.positive_words = {
            "bullish", "bull", "moon", "pump", "buy", "long", "profit", "gain",
            "up", "rise", "rising", "surge", "surging", "rally", "breakout",
            "ATH", "high", "higher", "growth", "growing", "strong", "strength",
            "positive", "optimistic", "confidence", "confident", "winning",
            "accumulate", "accumulating", "hodl", "hold", "support"
        }
        
        self.negative_words = {
            "bearish", "bear", "crash", "dump", "sell", "short", "loss", "lose",
            "down", "fall", "falling", "drop", "dropping", "plunge", "breakdown",
            "low", "lower", "weak", "weakness", "fear", "scared", "panic",
            "negative", "pessimistic", "worry", "worried", "losing",
            "distribute", "distributing", "resist", "resistance", "recession"
        }
        
        self.intensifiers = {
            "very", "extremely", "super", "mega", "ultra", "highly", "massive"
        }
        
        self.negators = {
            "not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't"
        }
    
    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze text sentiment.
        
        Returns:
            (sentiment_score, confidence)
            sentiment_score: -1 to +1
            confidence: 0 to 1
        """
        text = text.lower()
        words = re.findall(r'\w+', text)
        
        if not words:
            return 0, 0
        
        positive_count = 0
        negative_count = 0
        intensity_boost = 0
        
        prev_word = ""
        for word in words:
            # Check for negation
            negated = prev_word in self.negators
            
            if word in self.positive_words:
                if negated:
                    negative_count += 1
                else:
                    positive_count += 1
                    if prev_word in self.intensifiers:
                        intensity_boost += 0.5
            
            elif word in self.negative_words:
                if negated:
                    positive_count += 1
                else:
                    negative_count += 1
                    if prev_word in self.intensifiers:
                        intensity_boost += 0.5
            
            prev_word = word
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0, 0.1  # Neutral with low confidence
        
        # Calculate sentiment score
        raw_score = (positive_count - negative_count) / total_sentiment_words
        
        # Apply intensity boost
        sentiment_score = np.clip(raw_score + intensity_boost * np.sign(raw_score), -1, 1)
        
        # Confidence based on sample size and clarity
        clarity = abs(positive_count - negative_count) / total_sentiment_words
        sample_confidence = min(total_sentiment_words / 10, 1)
        confidence = clarity * 0.5 + sample_confidence * 0.5
        
        return sentiment_score, confidence


class FearGreedIndicator:
    """
    Fear & Greed Index calculation.
    Similar to CNN's Fear & Greed Index.
    """
    
    def __init__(self):
        self._components: Dict[str, deque] = {
            "momentum": deque(maxlen=100),
            "volatility": deque(maxlen=100),
            "volume": deque(maxlen=100),
            "social": deque(maxlen=100),
            "dominance": deque(maxlen=100)  # For crypto: BTC dominance
        }
        
        self._weights = {
            "momentum": 0.25,
            "volatility": 0.25,
            "volume": 0.15,
            "social": 0.20,
            "dominance": 0.15
        }
    
    def update_component(self, component: str, value: float):
        """Update a component value (0-100 scale)."""
        if component in self._components:
            self._components[component].append(value)
    
    def calculate(self) -> Tuple[float, str]:
        """
        Calculate Fear & Greed Index.
        
        Returns:
            (index, label)
            index: 0-100
            label: "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
        """
        total = 0
        weight_sum = 0
        
        for component, weight in self._weights.items():
            values = self._components[component]
            if values:
                total += np.mean(values) * weight
                weight_sum += weight
        
        if weight_sum == 0:
            return 50, "Neutral"
        
        index = total / weight_sum
        
        # Determine label
        if index < 20:
            label = "Extreme Fear"
        elif index < 40:
            label = "Fear"
        elif index < 60:
            label = "Neutral"
        elif index < 80:
            label = "Greed"
        else:
            label = "Extreme Greed"
        
        return index, label
    
    def get_contrarian_signal(self) -> Optional[str]:
        """
        Get contrarian signal if sentiment is extreme.
        Extreme fear = potential buy, Extreme greed = potential sell.
        """
        index, label = self.calculate()
        
        if index < 15:
            return "BUY"  # Extreme fear = contrarian buy
        elif index > 85:
            return "SELL"  # Extreme greed = contrarian sell
        
        return None


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources.
    """
    
    def __init__(self):
        self.text_analyzer = TextSentimentAnalyzer()
        self.fear_greed = FearGreedIndicator()
        
        # Source weights
        self.source_weights = {
            SentimentSource.SOCIAL: 0.20,
            SentimentSource.NEWS: 0.25,
            SentimentSource.FEAR_GREED: 0.25,
            SentimentSource.ONCHAIN: 0.15,
            SentimentSource.OPTIONS: 0.10,
            SentimentSource.ANALYST: 0.05
        }
        
        # Current readings
        self._readings: Dict[SentimentSource, SentimentReading] = {}
        
        # History for trend detection
        self._history = deque(maxlen=100)
    
    def add_text_sentiment(self, source: SentimentSource, texts: List[str]):
        """Analyze texts and add sentiment reading."""
        if not texts:
            return
        
        sentiments = []
        confidences = []
        
        for text in texts:
            score, conf = self.text_analyzer.analyze(text)
            sentiments.append(score)
            confidences.append(conf)
        
        avg_sentiment = np.mean(sentiments)
        avg_confidence = np.mean(confidences)
        
        self._readings[source] = SentimentReading(
            source=source,
            value=avg_sentiment,
            confidence=avg_confidence,
            sample_size=len(texts),
            metadata={"texts_analyzed": len(texts)}
        )
    
    def add_reading(self, reading: SentimentReading):
        """Add a sentiment reading directly."""
        self._readings[reading.source] = reading
    
    def update_fear_greed(
        self,
        momentum: float = None,
        volatility: float = None,
        volume: float = None,
        social: float = None,
        dominance: float = None
    ):
        """Update fear/greed indicator components."""
        if momentum is not None:
            self.fear_greed.update_component("momentum", momentum)
        if volatility is not None:
            # Invert volatility (high vol = fear)
            self.fear_greed.update_component("volatility", 100 - volatility)
        if volume is not None:
            self.fear_greed.update_component("volume", volume)
        if social is not None:
            self.fear_greed.update_component("social", social)
        if dominance is not None:
            self.fear_greed.update_component("dominance", dominance)
        
        # Update fear/greed reading
        index, label = self.fear_greed.calculate()
        
        # Convert 0-100 to -1 to +1
        normalized = (index - 50) / 50
        
        self._readings[SentimentSource.FEAR_GREED] = SentimentReading(
            source=SentimentSource.FEAR_GREED,
            value=normalized,
            confidence=0.8,  # Generally reliable
            sample_size=1,
            metadata={"index": index, "label": label}
        )
    
    def aggregate(self) -> AggregateSentiment:
        """Aggregate all sentiment sources."""
        if not self._readings:
            return AggregateSentiment(
                overall=0,
                confidence=0,
                readings={},
                contrarian_signal=False,
                trend="stable"
            )
        
        # Weighted average
        total_sentiment = 0
        total_weight = 0
        total_confidence = 0
        
        for source, reading in self._readings.items():
            weight = self.source_weights.get(source, 0.1)
            total_sentiment += reading.value * weight * reading.confidence
            total_weight += weight * reading.confidence
            total_confidence += reading.confidence
        
        overall = total_sentiment / total_weight if total_weight > 0 else 0
        confidence = total_confidence / len(self._readings)
        
        # Add to history
        self._history.append(overall)
        
        # Determine trend
        trend = self._determine_trend()
        
        # Check for contrarian signal
        contrarian = abs(overall) > 0.7
        
        return AggregateSentiment(
            overall=overall,
            confidence=confidence,
            readings={s.value: r for s, r in self._readings.items()},
            contrarian_signal=contrarian,
            trend=trend
        )
    
    def _determine_trend(self) -> str:
        """Determine sentiment trend."""
        if len(self._history) < 5:
            return "stable"
        
        recent = np.array(list(self._history)[-5:])
        older = np.array(list(self._history)[-20:-5]) if len(self._history) >= 20 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        diff = recent_avg - older_avg
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "worsening"
        return "stable"
    
    def get_features(self) -> Dict[str, float]:
        """Get sentiment features for ML models."""
        agg = self.aggregate()
        
        features = {
            "sentiment_overall": agg.overall,
            "sentiment_confidence": agg.confidence,
            "sentiment_contrarian": 1.0 if agg.contrarian_signal else 0.0,
            "sentiment_improving": 1.0 if agg.trend == "improving" else 0.0,
            "sentiment_worsening": 1.0 if agg.trend == "worsening" else 0.0
        }
        
        # Add individual source readings
        for source in SentimentSource:
            reading = self._readings.get(source)
            if reading:
                features[f"sentiment_{source.value}"] = reading.value
                features[f"sentiment_{source.value}_conf"] = reading.confidence
            else:
                features[f"sentiment_{source.value}"] = 0.0
                features[f"sentiment_{source.value}_conf"] = 0.0
        
        # Fear/greed specific
        fg_reading = self._readings.get(SentimentSource.FEAR_GREED)
        if fg_reading and "index" in fg_reading.metadata:
            features["fear_greed_index"] = fg_reading.metadata["index"] / 100
        else:
            features["fear_greed_index"] = 0.5
        
        return features


class SentimentEngine:
    """
    Master sentiment engine that can fetch and aggregate sentiment.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.aggregator = SentimentAggregator()
        
        # Cache
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)
    
    async def fetch_all(self, symbol: str = None):
        """Fetch sentiment from all available sources."""
        tasks = []
        
        # Add fetch tasks based on available APIs
        # (In production, these would call real APIs)
        
        # For now, simulate with cached/default values
        self._simulate_sentiment(symbol)
    
    def _simulate_sentiment(self, symbol: str = None):
        """Simulate sentiment data for testing."""
        import random
        
        # Simulate social sentiment
        self.aggregator.add_reading(SentimentReading(
            source=SentimentSource.SOCIAL,
            value=random.uniform(-0.3, 0.3),
            confidence=0.6,
            sample_size=100
        ))
        
        # Simulate news sentiment
        self.aggregator.add_reading(SentimentReading(
            source=SentimentSource.NEWS,
            value=random.uniform(-0.2, 0.2),
            confidence=0.7,
            sample_size=20
        ))
        
        # Update fear/greed
        self.aggregator.update_fear_greed(
            momentum=random.uniform(30, 70),
            volatility=random.uniform(20, 60),
            social=random.uniform(40, 60)
        )
    
    def get_sentiment(self) -> AggregateSentiment:
        """Get current aggregated sentiment."""
        return self.aggregator.aggregate()
    
    def get_features(self) -> Dict[str, float]:
        """Get sentiment features for ML models."""
        return self.aggregator.get_features()
    
    def get_trading_signal(self) -> Optional[Tuple[str, float]]:
        """
        Get trading signal from sentiment.
        
        Returns:
            (direction, confidence) or None
        """
        agg = self.aggregator.aggregate()
        
        # Contrarian logic for extreme sentiment
        if agg.contrarian_signal:
            if agg.overall > 0.7:
                return ("SELL", abs(agg.overall) * 0.5)  # Extreme greed = sell
            elif agg.overall < -0.7:
                return ("BUY", abs(agg.overall) * 0.5)  # Extreme fear = buy
        
        # Normal sentiment following
        if abs(agg.overall) > 0.3 and agg.confidence > 0.5:
            direction = "BUY" if agg.overall > 0 else "SELL"
            return (direction, abs(agg.overall) * agg.confidence)
        
        return None

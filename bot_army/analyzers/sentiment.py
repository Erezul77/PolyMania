"""Sentiment analysis for market intelligence."""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("analyzer.sentiment")


@dataclass
class SentimentScore:
    """Sentiment analysis result."""
    source: str
    text: str
    score: float  # -1 to 1
    magnitude: float  # 0 to 1
    keywords: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def label(self) -> str:
        if self.score > 0.2:
            return "POSITIVE"
        elif self.score < -0.2:
            return "NEGATIVE"
        return "NEUTRAL"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "text": self.text[:100],
            "score": self.score,
            "magnitude": self.magnitude,
            "label": self.label,
            "keywords": self.keywords,
            "timestamp": self.timestamp.isoformat()
        }


class SentimentAnalyzer:
    """
    Multi-source sentiment analyzer.
    Analyzes text from news, social media, and market commentary.
    """
    
    # Sentiment lexicon
    POSITIVE_WORDS = {
        "bullish", "positive", "up", "rise", "gain", "growth", "surge",
        "rally", "strong", "confident", "optimistic", "win", "success",
        "higher", "increase", "boost", "recover", "breakthrough", "soar",
        "outperform", "beat", "exceed", "improve", "advance", "momentum"
    }
    
    NEGATIVE_WORDS = {
        "bearish", "negative", "down", "fall", "loss", "decline", "drop",
        "crash", "weak", "worried", "pessimistic", "lose", "fail",
        "lower", "decrease", "cut", "collapse", "plunge", "slump",
        "underperform", "miss", "disappoint", "concern", "risk", "fear"
    }
    
    # Intensity modifiers
    INTENSIFIERS = {
        "very": 1.5, "extremely": 2.0, "highly": 1.5, "strongly": 1.5,
        "significantly": 1.5, "massively": 2.0, "slightly": 0.5,
        "somewhat": 0.7, "marginally": 0.5, "incredibly": 2.0
    }
    
    NEGATORS = {"not", "no", "never", "none", "neither", "without", "hardly"}
    
    def __init__(self):
        self._cache: Dict[str, SentimentScore] = {}
        self._market_sentiment: Dict[str, List[SentimentScore]] = defaultdict(list)
        self._history_size = 100
    
    def analyze_text(
        self,
        text: str,
        source: str = "unknown"
    ) -> SentimentScore:
        """Analyze sentiment of text."""
        # Preprocess
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Calculate sentiment
        positive_count = 0
        negative_count = 0
        intensity = 1.0
        keywords = []
        
        negated = False
        
        for i, word in enumerate(words):
            # Check for negation
            if word in self.NEGATORS:
                negated = True
                continue
            
            # Check for intensity modifier
            if word in self.INTENSIFIERS:
                intensity = self.INTENSIFIERS[word]
                continue
            
            # Score sentiment
            if word in self.POSITIVE_WORDS:
                keywords.append(word)
                if negated:
                    negative_count += intensity
                    negated = False
                else:
                    positive_count += intensity
                intensity = 1.0
            
            elif word in self.NEGATIVE_WORDS:
                keywords.append(word)
                if negated:
                    positive_count += intensity
                    negated = False
                else:
                    negative_count += intensity
                intensity = 1.0
        
        # Calculate final score
        total = positive_count + negative_count
        if total > 0:
            score = (positive_count - negative_count) / total
            magnitude = min(1.0, total / 10)
        else:
            score = 0.0
            magnitude = 0.0
        
        result = SentimentScore(
            source=source,
            text=text[:500],
            score=score,
            magnitude=magnitude,
            keywords=keywords[:10]
        )
        
        self._cache[text[:100]] = result
        return result
    
    def analyze_market(
        self,
        market_id: str,
        texts: List[Tuple[str, str]]  # (text, source) pairs
    ) -> Dict[str, Any]:
        """Analyze sentiment for a specific market."""
        scores = []
        
        for text, source in texts:
            score = self.analyze_text(text, source)
            scores.append(score)
            self._market_sentiment[market_id].append(score)
        
        # Trim history
        self._market_sentiment[market_id] = \
            self._market_sentiment[market_id][-self._history_size:]
        
        if not scores:
            return {
                "market_id": market_id,
                "score": 0.0,
                "magnitude": 0.0,
                "label": "NEUTRAL",
                "sample_count": 0
            }
        
        # Aggregate
        avg_score = sum(s.score * s.magnitude for s in scores) / sum(s.magnitude for s in scores) if sum(s.magnitude for s in scores) > 0 else 0
        avg_magnitude = sum(s.magnitude for s in scores) / len(scores)
        
        # Weighted by recency
        weights = [0.5 ** i for i in range(len(scores))]
        weighted_score = sum(s.score * w for s, w in zip(reversed(scores), weights)) / sum(weights)
        
        return {
            "market_id": market_id,
            "score": weighted_score,
            "magnitude": avg_magnitude,
            "label": "POSITIVE" if weighted_score > 0.2 else "NEGATIVE" if weighted_score < -0.2 else "NEUTRAL",
            "sample_count": len(scores),
            "positive_count": sum(1 for s in scores if s.score > 0.2),
            "negative_count": sum(1 for s in scores if s.score < -0.2),
            "keywords": list(set(kw for s in scores for kw in s.keywords))[:20]
        }
    
    def get_market_sentiment_history(
        self,
        market_id: str,
        minutes: int = 60
    ) -> List[SentimentScore]:
        """Get sentiment history for market."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            s for s in self._market_sentiment.get(market_id, [])
            if s.timestamp > cutoff
        ]
    
    def detect_sentiment_shift(
        self,
        market_id: str,
        window_minutes: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Detect significant sentiment shifts."""
        history = self.get_market_sentiment_history(market_id, window_minutes * 2)
        
        if len(history) < 5:
            return None
        
        midpoint = len(history) // 2
        old_scores = [s.score for s in history[:midpoint]]
        new_scores = [s.score for s in history[midpoint:]]
        
        old_avg = sum(old_scores) / len(old_scores) if old_scores else 0
        new_avg = sum(new_scores) / len(new_scores) if new_scores else 0
        
        shift = new_avg - old_avg
        
        if abs(shift) > 0.3:
            return {
                "market_id": market_id,
                "shift": shift,
                "direction": "IMPROVING" if shift > 0 else "DETERIORATING",
                "old_sentiment": old_avg,
                "new_sentiment": new_avg,
                "significance": abs(shift) / 0.3  # >1 means significant
            }
        
        return None
    
    def analyze_batch(
        self,
        items: List[Dict[str, str]]
    ) -> List[SentimentScore]:
        """Analyze multiple items."""
        return [
            self.analyze_text(item["text"], item.get("source", "unknown"))
            for item in items
        ]
    
    def get_aggregate_sentiment(
        self,
        minutes: int = 60
    ) -> Dict[str, Any]:
        """Get aggregate market sentiment."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        all_scores = []
        for market_scores in self._market_sentiment.values():
            all_scores.extend([s for s in market_scores if s.timestamp > cutoff])
        
        if not all_scores:
            return {
                "score": 0.0,
                "label": "NEUTRAL",
                "sample_count": 0
            }
        
        avg_score = sum(s.score for s in all_scores) / len(all_scores)
        
        return {
            "score": avg_score,
            "label": "POSITIVE" if avg_score > 0.2 else "NEGATIVE" if avg_score < -0.2 else "NEUTRAL",
            "sample_count": len(all_scores),
            "positive_pct": sum(1 for s in all_scores if s.score > 0.2) / len(all_scores) * 100,
            "negative_pct": sum(1 for s in all_scores if s.score < -0.2) / len(all_scores) * 100,
            "markets_analyzed": len(self._market_sentiment)
        }

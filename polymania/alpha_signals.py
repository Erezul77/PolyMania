"""
Alpha Signal Processor - External Intelligence Integration

This module connects external signals (Telegram, News) to trading decisions,
giving us an edge over pure technical analysis bots.

Key Concepts:
1. SPEED EDGE: React to external signals BEFORE they hit prices
2. INFORMATION EDGE: Use signals other bots don't have access to
3. CONTRARIAN EDGE: Be skeptical when everyone's on the same side
"""

import csv
import os
import re
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from .news_client import search_news
from .correlation import TOPIC_GROUPS, detect_topic_for_event

logger = logging.getLogger('polymania.alpha')

# Files
TELEGRAM_HITS_CSV = 'data/telegram_hits.csv'
ALPHA_SIGNALS_CSV = 'data/alpha_signals.csv'

# How recent a Telegram hit needs to be to boost a signal (seconds)
TELEGRAM_FRESHNESS_SEC = 600  # 10 minutes
NEWS_FRESHNESS_SEC = 3600  # 1 hour


class SentimentType(Enum):
    VERY_BULLISH = 'VERY_BULLISH'
    BULLISH = 'BULLISH'
    NEUTRAL = 'NEUTRAL'
    BEARISH = 'BEARISH'
    VERY_BEARISH = 'VERY_BEARISH'


@dataclass
class AlphaSignal:
    """External signal that can boost/reduce trading confidence"""
    timestamp: datetime
    source: str  # 'telegram', 'news', 'correlation'
    topic: Optional[str]
    sentiment: SentimentType
    confidence_adjustment: float  # -0.3 to +0.3
    matched_keywords: List[str] = field(default_factory=list)
    raw_text: str = ''
    event_match_score: float = 0.0  # How well this matches a Polymarket event
    reasons: List[str] = field(default_factory=list)


# Sentiment keywords for different topics
SENTIMENT_KEYWORDS = {
    'bullish': [
        'surge', 'soar', 'rally', 'breakout', 'moon', 'pump', 'bullish',
        'win', 'wins', 'winning', 'victory', 'success', 'confirmed',
        'approved', 'passed', 'positive', 'strong', 'higher', 'up',
        'breakthrough', 'deal', 'agreement', 'peace', 'ceasefire agreed',
    ],
    'bearish': [
        'crash', 'dump', 'plunge', 'bearish', 'collapse', 'fail',
        'loss', 'loses', 'losing', 'defeat', 'rejected', 'denied',
        'negative', 'weak', 'lower', 'down', 'crisis', 'war',
        'attack', 'strikes', 'invasion', 'conflict', 'escalation',
    ],
    'uncertainty': [
        'rumor', 'unconfirmed', 'alleged', 'possibly', 'maybe',
        'speculation', 'uncertain', 'unknown', 'developing',
    ]
}

# Keywords that indicate HIGH URGENCY (act fast!)
URGENCY_KEYWORDS = [
    'breaking', 'just in', 'urgent', 'flash', 'alert',
    'happening now', 'live', 'confirmed', 'official',
]


def analyze_sentiment(text: str) -> Tuple[SentimentType, float, List[str]]:
    """
    Analyze text sentiment and return (sentiment, confidence, reasons)
    """
    text_lower = text.lower()
    reasons = []
    
    bullish_count = sum(1 for kw in SENTIMENT_KEYWORDS['bullish'] if kw in text_lower)
    bearish_count = sum(1 for kw in SENTIMENT_KEYWORDS['bearish'] if kw in text_lower)
    uncertainty_count = sum(1 for kw in SENTIMENT_KEYWORDS['uncertainty'] if kw in text_lower)
    urgency_count = sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)
    
    # Calculate net sentiment
    net = bullish_count - bearish_count
    total = bullish_count + bearish_count
    
    if total == 0:
        return SentimentType.NEUTRAL, 0.0, ['No sentiment keywords found']
    
    # Base confidence from keyword density
    confidence = min(0.3, total * 0.05)
    
    # Reduce confidence if uncertain language
    if uncertainty_count > 0:
        confidence *= 0.5
        reasons.append(f'Uncertainty detected ({uncertainty_count} keywords)')
    
    # Boost confidence if urgent
    if urgency_count > 0:
        confidence = min(0.3, confidence * 1.5)
        reasons.append(f'URGENT signal ({urgency_count} keywords)')
    
    # Determine sentiment type
    if net >= 3:
        sentiment = SentimentType.VERY_BULLISH
        reasons.append(f'Strong bullish ({bullish_count} bullish vs {bearish_count} bearish)')
    elif net >= 1:
        sentiment = SentimentType.BULLISH
        reasons.append(f'Bullish ({bullish_count} vs {bearish_count})')
    elif net <= -3:
        sentiment = SentimentType.VERY_BEARISH
        reasons.append(f'Strong bearish ({bearish_count} bearish vs {bullish_count} bullish)')
    elif net <= -1:
        sentiment = SentimentType.BEARISH
        reasons.append(f'Bearish ({bearish_count} vs {bullish_count})')
    else:
        sentiment = SentimentType.NEUTRAL
        reasons.append('Mixed sentiment')
    
    return sentiment, confidence, reasons


def match_to_polymarket_event(text: str, event_title: str, event_id: str = '') -> float:
    """
    Calculate how well external signal text matches a Polymarket event.
    Returns score 0.0 to 1.0
    """
    text_lower = text.lower()
    title_lower = event_title.lower()
    
    # Extract key terms from event title
    # Remove common words
    stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'vs', 'vs.', 'will', 'be', 'by'}
    title_words = set(title_lower.split()) - stop_words
    
    if not title_words:
        return 0.0
    
    # Count matches
    matches = sum(1 for word in title_words if word in text_lower and len(word) > 2)
    score = matches / len(title_words)
    
    # Boost if exact phrases match
    # Extract potential entity names (capitalized sequences in original title)
    entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', event_title)
    for entity in entities:
        if entity.lower() in text_lower:
            score = min(1.0, score + 0.2)
    
    return score


def get_recent_telegram_hits(max_age_sec: int = TELEGRAM_FRESHNESS_SEC) -> List[Dict[str, Any]]:
    """
    Read recent Telegram hits from CSV
    """
    hits = []
    
    if not os.path.exists(TELEGRAM_HITS_CSV):
        return hits
    
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_sec)
    
    try:
        with open(TELEGRAM_HITS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Parse timestamp
                    ts_str = row.get('scan_timestamp_utc', '')
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        
                        if ts >= cutoff:
                            hits.append({
                                'timestamp': ts,
                                'channel': row.get('channel_title', ''),
                                'keyword': row.get('matched_keyword', ''),
                                'text': row.get('text_excerpt', ''),
                                'url': row.get('message_url', ''),
                            })
                except Exception as e:
                    logger.debug(f'Error parsing telegram hit: {e}')
    except Exception as e:
        logger.error(f'Error reading telegram hits: {e}')
    
    return hits


def get_news_for_event(event_title: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search news for an event and analyze sentiment
    """
    # Extract search query from title
    # Remove common suffixes like "More Markets", "on January 5"
    query = re.sub(r'\s*-\s*More Markets.*$', '', event_title)
    query = re.sub(r'\s+on\s+\w+\s+\d+.*$', '', query)
    query = query[:60]  # Limit query length
    
    return search_news(query, max_results)


def generate_alpha_signals(event_title: str, event_id: str = '') -> List[AlphaSignal]:
    """
    Generate alpha signals from external sources for a given event.
    
    Returns list of AlphaSignal objects that can be used to adjust trading confidence.
    """
    signals = []
    
    # 1. Check Telegram hits
    telegram_hits = get_recent_telegram_hits()
    for hit in telegram_hits:
        match_score = match_to_polymarket_event(hit['text'], event_title, event_id)
        
        if match_score >= 0.3:  # Minimum match threshold
            sentiment, conf_adj, reasons = analyze_sentiment(hit['text'])
            
            signal = AlphaSignal(
                timestamp=hit['timestamp'],
                source='telegram',
                topic=detect_topic_for_event(event_title, ''),
                sentiment=sentiment,
                confidence_adjustment=conf_adj if sentiment in [SentimentType.BULLISH, SentimentType.VERY_BULLISH] else -conf_adj,
                matched_keywords=[hit['keyword']],
                raw_text=hit['text'][:200],
                event_match_score=match_score,
                reasons=reasons + [f'Telegram match: {match_score:.0%}', f'Channel: {hit["channel"]}']
            )
            signals.append(signal)
            logger.info(f'Alpha signal from Telegram: {sentiment.value} for {event_title[:40]}')
    
    # 2. Check News (if we have API key configured)
    try:
        news_articles = get_news_for_event(event_title)
        for article in news_articles:
            title = article.get('title', '') or ''
            
            # Check freshness
            pub_date_str = article.get('publishedAt', '')
            if pub_date_str:
                try:
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                    age = (datetime.now(timezone.utc) - pub_date).total_seconds()
                    if age > NEWS_FRESHNESS_SEC:
                        continue
                except:
                    pass
            
            sentiment, conf_adj, reasons = analyze_sentiment(title)
            
            if sentiment != SentimentType.NEUTRAL:
                signal = AlphaSignal(
                    timestamp=datetime.now(timezone.utc),
                    source='news',
                    topic=detect_topic_for_event(event_title, ''),
                    sentiment=sentiment,
                    confidence_adjustment=conf_adj if sentiment in [SentimentType.BULLISH, SentimentType.VERY_BULLISH] else -conf_adj,
                    matched_keywords=[],
                    raw_text=title[:200],
                    event_match_score=1.0,  # Direct news search
                    reasons=reasons + [f'Source: {article.get("source", "Unknown")}']
                )
                signals.append(signal)
                logger.debug(f'Alpha signal from News: {sentiment.value} for {event_title[:40]}')
    except Exception as e:
        logger.debug(f'News search failed: {e}')
    
    return signals


def calculate_alpha_adjustment(signals: List[AlphaSignal], base_signal_type: str) -> Tuple[float, List[str]]:
    """
    Calculate overall confidence adjustment from alpha signals.
    
    Args:
        signals: List of AlphaSignal objects
        base_signal_type: The base signal type ('BUY', 'SELL', etc.)
    
    Returns:
        (adjustment, reasons) where adjustment is -0.3 to +0.3
    """
    if not signals:
        return 0.0, []
    
    reasons = []
    total_adjustment = 0.0
    
    is_buy_signal = 'BUY' in base_signal_type.upper()
    is_sell_signal = 'SELL' in base_signal_type.upper()
    
    for signal in signals:
        # Bullish alpha + BUY signal = boost
        # Bullish alpha + SELL signal = reduce
        # Bearish alpha + SELL signal = boost
        # Bearish alpha + BUY signal = reduce
        
        if signal.sentiment in [SentimentType.BULLISH, SentimentType.VERY_BULLISH]:
            if is_buy_signal:
                total_adjustment += signal.confidence_adjustment
                reasons.append(f'Alpha BOOST: {signal.source} bullish signal (+{signal.confidence_adjustment:.0%})')
            elif is_sell_signal:
                total_adjustment -= signal.confidence_adjustment * 0.5  # Be cautious
                reasons.append(f'Alpha CAUTION: {signal.source} bullish but selling')
        
        elif signal.sentiment in [SentimentType.BEARISH, SentimentType.VERY_BEARISH]:
            if is_sell_signal:
                total_adjustment += abs(signal.confidence_adjustment)
                reasons.append(f'Alpha BOOST: {signal.source} bearish signal (+{abs(signal.confidence_adjustment):.0%})')
            elif is_buy_signal:
                total_adjustment -= abs(signal.confidence_adjustment) * 0.5
                reasons.append(f'Alpha CAUTION: {signal.source} bearish but buying')
    
    # Cap adjustment
    total_adjustment = max(-0.3, min(0.3, total_adjustment))
    
    return total_adjustment, reasons


def log_alpha_signal(signal: AlphaSignal, event_id: str, event_title: str):
    """Log alpha signal to CSV for analysis"""
    os.makedirs(os.path.dirname(ALPHA_SIGNALS_CSV) or '.', exist_ok=True)
    
    headers = ['timestamp', 'event_id', 'event_title', 'source', 'sentiment', 
               'adjustment', 'match_score', 'keywords', 'reasons']
    
    if not os.path.exists(ALPHA_SIGNALS_CSV) or os.path.getsize(ALPHA_SIGNALS_CSV) == 0:
        with open(ALPHA_SIGNALS_CSV, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)
    
    row = [
        signal.timestamp.isoformat(),
        event_id,
        event_title[:50],
        signal.source,
        signal.sentiment.value,
        round(signal.confidence_adjustment, 3),
        round(signal.event_match_score, 2),
        '|'.join(signal.matched_keywords),
        '|'.join(signal.reasons),
    ]
    
    with open(ALPHA_SIGNALS_CSV, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)


# Singleton for caching
_alpha_cache: Dict[str, Tuple[List[AlphaSignal], datetime]] = {}
CACHE_TTL_SEC = 60


def get_alpha_signals_cached(event_title: str, event_id: str = '') -> List[AlphaSignal]:
    """Get alpha signals with caching to avoid repeated API calls"""
    cache_key = event_id or event_title[:50]
    now = datetime.now(timezone.utc)
    
    if cache_key in _alpha_cache:
        signals, cached_at = _alpha_cache[cache_key]
        if (now - cached_at).total_seconds() < CACHE_TTL_SEC:
            return signals
    
    signals = generate_alpha_signals(event_title, event_id)
    _alpha_cache[cache_key] = (signals, now)
    
    # Log signals
    for signal in signals:
        log_alpha_signal(signal, event_id, event_title)
    
    return signals

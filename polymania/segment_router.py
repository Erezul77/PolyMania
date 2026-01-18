"""
Segment Router - Route signals to segment-specific strategies

Different market segments require different approaches:
- CRYPTO: High volatility, 24/7, whale-driven → wider stops, momentum
- SPORTS: Binary outcomes, event-driven → quick entries, timing critical
- POLITICS: Sentiment-driven, longer holds → news alpha, contrarian
- WEATHER: Data-predictable → model-based, wait for mispricings
- ESPORTS: Fast-moving, insider knowledge → follow smart money
- OTHER: Mixed bag → balanced approach

Each segment gets optimized parameters based on historical performance.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger('polymania.segment_router')


class MarketSegment(Enum):
    CRYPTO = 'crypto'
    SPORTS = 'sports'
    POLITICS = 'politics'
    WEATHER = 'weather'
    ESPORTS = 'esports'
    ENTERTAINMENT = 'entertainment'
    SCIENCE = 'science'
    OTHER = 'other'


@dataclass
class SegmentConfig:
    """Configuration for a specific market segment"""
    segment: MarketSegment
    
    # Signal thresholds
    min_confidence: float = 0.5  # Minimum confidence to trade
    strong_signal_threshold: float = 0.75  # Confidence for "strong" signals
    
    # Position sizing
    base_position_pct: float = 5.0  # % of portfolio per trade
    max_position_pct: float = 10.0  # Maximum position size
    
    # Stop loss / Take profit
    stop_loss_pct: float = 8.0  # Default stop loss %
    take_profit_pct: float = 15.0  # Default take profit %
    
    # Timing
    min_time_to_resolution_hours: float = 2.0  # Don't enter too close to resolution
    
    # Technical analysis weights
    rsi_weight: float = 1.0
    momentum_weight: float = 1.0
    trend_weight: float = 1.0
    
    # Alpha signal weights
    telegram_alpha_weight: float = 1.0
    whale_signal_weight: float = 1.0
    news_weight: float = 1.0
    
    # Contrarian settings
    contrarian_enabled: bool = True
    overbought_threshold: float = 75.0
    oversold_threshold: float = 25.0
    
    # Special flags
    follow_momentum: bool = True
    fade_extremes: bool = True
    prioritize_alpha: bool = False


# Segment-specific configurations based on our learnings
SEGMENT_CONFIGS: Dict[MarketSegment, SegmentConfig] = {
    MarketSegment.CRYPTO: SegmentConfig(
        segment=MarketSegment.CRYPTO,
        min_confidence=0.55,  # Higher bar - crypto is noisy
        strong_signal_threshold=0.8,
        base_position_pct=3.0,  # Smaller positions - volatile
        max_position_pct=6.0,
        stop_loss_pct=12.0,  # WIDER stops - crypto swings
        take_profit_pct=20.0,
        rsi_weight=0.8,  # RSI less reliable in crypto
        momentum_weight=1.5,  # Momentum matters more
        whale_signal_weight=2.0,  # Whales REALLY matter in crypto
        contrarian_enabled=True,
        overbought_threshold=80.0,  # Higher threshold - crypto runs further
        oversold_threshold=20.0,
        follow_momentum=True,
        prioritize_alpha=True,  # External signals crucial
    ),
    
    MarketSegment.SPORTS: SegmentConfig(
        segment=MarketSegment.SPORTS,
        min_confidence=0.5,
        strong_signal_threshold=0.7,
        base_position_pct=5.0,
        max_position_pct=8.0,
        stop_loss_pct=5.0,  # Tighter stops - binary outcomes
        take_profit_pct=10.0,
        min_time_to_resolution_hours=1.0,  # Can trade closer to event
        rsi_weight=0.5,  # RSI less relevant for sports
        momentum_weight=1.0,
        telegram_alpha_weight=1.5,  # Insider tips matter
        whale_signal_weight=1.5,
        contrarian_enabled=False,  # Less contrarian in sports
        follow_momentum=True,
    ),
    
    MarketSegment.POLITICS: SegmentConfig(
        segment=MarketSegment.POLITICS,
        min_confidence=0.5,
        strong_signal_threshold=0.7,
        base_position_pct=5.0,
        max_position_pct=10.0,
        stop_loss_pct=8.0,
        take_profit_pct=15.0,
        min_time_to_resolution_hours=24.0,  # Don't trade last day
        rsi_weight=1.0,
        momentum_weight=0.8,
        news_weight=2.0,  # News CRUCIAL for politics
        telegram_alpha_weight=2.0,
        contrarian_enabled=True,
        overbought_threshold=70.0,
        oversold_threshold=30.0,
        fade_extremes=True,  # Political markets overreact
        prioritize_alpha=True,
    ),
    
    MarketSegment.WEATHER: SegmentConfig(
        segment=MarketSegment.WEATHER,
        min_confidence=0.45,  # Can be more confident with weather
        strong_signal_threshold=0.65,
        base_position_pct=6.0,  # Larger positions - more predictable
        max_position_pct=12.0,
        stop_loss_pct=10.0,
        take_profit_pct=12.0,
        rsi_weight=0.3,  # Technical less relevant
        momentum_weight=0.5,
        contrarian_enabled=False,  # Weather is data-driven
        follow_momentum=False,  # Don't chase weather
        fade_extremes=False,
    ),
    
    MarketSegment.ESPORTS: SegmentConfig(
        segment=MarketSegment.ESPORTS,
        min_confidence=0.5,
        strong_signal_threshold=0.7,
        base_position_pct=4.0,
        max_position_pct=8.0,
        stop_loss_pct=6.0,
        take_profit_pct=12.0,
        min_time_to_resolution_hours=0.5,  # Very close to event OK
        rsi_weight=0.7,
        momentum_weight=1.2,
        whale_signal_weight=1.8,  # Insiders know teams
        contrarian_enabled=True,
        follow_momentum=True,
    ),
    
    MarketSegment.ENTERTAINMENT: SegmentConfig(
        segment=MarketSegment.ENTERTAINMENT,
        min_confidence=0.5,
        strong_signal_threshold=0.7,
        base_position_pct=4.0,
        max_position_pct=8.0,
        stop_loss_pct=8.0,
        take_profit_pct=15.0,
        news_weight=1.5,
        telegram_alpha_weight=1.5,
        contrarian_enabled=True,
    ),
    
    MarketSegment.SCIENCE: SegmentConfig(
        segment=MarketSegment.SCIENCE,
        min_confidence=0.55,
        strong_signal_threshold=0.75,
        base_position_pct=5.0,
        max_position_pct=10.0,
        stop_loss_pct=10.0,
        take_profit_pct=20.0,
        news_weight=2.0,
        contrarian_enabled=False,
    ),
    
    MarketSegment.OTHER: SegmentConfig(
        segment=MarketSegment.OTHER,
        # Default balanced config - this is what's been working best!
        min_confidence=0.5,
        strong_signal_threshold=0.7,
        base_position_pct=5.0,
        max_position_pct=10.0,
        stop_loss_pct=8.0,  # SL8 has been winning
        take_profit_pct=15.0,
        contrarian_enabled=True,
        follow_momentum=True,
    ),
}


# Keywords for segment detection
SEGMENT_KEYWORDS = {
    MarketSegment.CRYPTO: [
        'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'xrp', 
        'crypto', 'token', 'coin', 'blockchain', 'defi', 'nft',
        'binance', 'coinbase', 'price above', 'price below', 'price on'
    ],
    MarketSegment.SPORTS: [
        'vs.', 'vs ', 'match', 'game', 'league', 'cup', 'championship',
        'fc', 'united', 'city', 'team', 'nba', 'nfl', 'mlb', 'nhl',
        'premier league', 'la liga', 'serie a', 'bundesliga',
        'win', 'score', 'goals', 'points'
    ],
    MarketSegment.POLITICS: [
        'trump', 'biden', 'election', 'president', 'congress', 'senate',
        'vote', 'primary', 'democrat', 'republican', 'governor',
        'poll', 'approval', 'impeach', 'legislation', 'bill'
    ],
    MarketSegment.WEATHER: [
        'temperature', 'weather', 'rain', 'snow', 'hurricane', 'storm',
        'celsius', 'fahrenheit', 'degrees', 'highest', 'lowest',
        'forecast', 'climate'
    ],
    MarketSegment.ESPORTS: [
        'esports', 'dota', 'lol', 'league of legends', 'counter-strike',
        'cs2', 'csgo', 'valorant', 'overwatch', 'fortnite',
        'gaming', 'twitch', 'bo3', 'bo5', 'tournament'
    ],
    MarketSegment.ENTERTAINMENT: [
        'oscar', 'grammy', 'emmy', 'movie', 'film', 'album', 'song',
        'netflix', 'disney', 'spotify', 'youtube', 'tiktok',
        'celebrity', 'actor', 'singer', 'artist'
    ],
    MarketSegment.SCIENCE: [
        'spacex', 'nasa', 'rocket', 'launch', 'mars', 'moon',
        'ai', 'artificial intelligence', 'gpt', 'openai',
        'research', 'discovery', 'breakthrough'
    ],
}


def detect_segment(event_title: str, event_slug: str = '') -> MarketSegment:
    """
    Detect the market segment from event title and slug.
    """
    text = f"{event_title} {event_slug}".lower()
    
    # Check each segment's keywords
    scores = {segment: 0 for segment in MarketSegment}
    
    for segment, keywords in SEGMENT_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text:
                scores[segment] += 1
                # Bonus for longer/more specific keywords
                if len(kw) > 5:
                    scores[segment] += 0.5
    
    # Find best match
    best_segment = max(scores, key=scores.get)
    
    if scores[best_segment] > 0:
        return best_segment
    
    return MarketSegment.OTHER


def get_segment_config(segment: MarketSegment) -> SegmentConfig:
    """Get configuration for a segment"""
    return SEGMENT_CONFIGS.get(segment, SEGMENT_CONFIGS[MarketSegment.OTHER])


def get_config_for_event(event_title: str, event_slug: str = '') -> Tuple[MarketSegment, SegmentConfig]:
    """
    Detect segment and return appropriate config for an event.
    """
    segment = detect_segment(event_title, event_slug)
    config = get_segment_config(segment)
    return segment, config


def adjust_signal_for_segment(
    signal_confidence: float,
    signal_type: str,
    event_title: str,
    rsi: Optional[float] = None,
    has_alpha: bool = False,
    has_whale: bool = False,
) -> Tuple[float, str, List[str]]:
    """
    Adjust signal confidence based on segment-specific rules.
    
    Returns:
        (adjusted_confidence, action, reasons)
    """
    segment, config = get_config_for_event(event_title)
    reasons = [f'Segment: {segment.value}']
    
    adjusted = signal_confidence
    action = 'TRADE'
    
    # Check minimum confidence
    if signal_confidence < config.min_confidence:
        action = 'SKIP'
        reasons.append(f'Below min confidence ({config.min_confidence:.0%})')
        return adjusted, action, reasons
    
    # Apply RSI-based adjustments
    if rsi is not None and config.contrarian_enabled:
        if rsi > config.overbought_threshold:
            if 'BUY' in signal_type.upper():
                adjusted -= 0.1
                reasons.append(f'RSI {rsi:.0f} > {config.overbought_threshold} overbought')
                if config.fade_extremes and rsi > 85:
                    action = 'SKIP'
                    reasons.append('Fading extreme - skip buy')
            elif 'SELL' in signal_type.upper() and config.fade_extremes:
                adjusted += 0.1
                reasons.append('Contrarian sell at overbought')
        
        elif rsi < config.oversold_threshold:
            if 'SELL' in signal_type.upper():
                adjusted -= 0.1
                reasons.append(f'RSI {rsi:.0f} < {config.oversold_threshold} oversold')
                if config.fade_extremes and rsi < 15:
                    action = 'SKIP'
                    reasons.append('Fading extreme - skip sell')
            elif 'BUY' in signal_type.upper() and config.fade_extremes:
                adjusted += 0.1
                reasons.append('Contrarian buy at oversold')
    
    # Apply alpha signal boosts
    if has_alpha and config.prioritize_alpha:
        adjusted += 0.1 * config.telegram_alpha_weight
        reasons.append('Alpha signal boost')
    
    if has_whale:
        adjusted += 0.05 * config.whale_signal_weight
        reasons.append('Whale signal boost')
    
    # Cap confidence
    adjusted = max(0.1, min(0.95, adjusted))
    
    return adjusted, action, reasons


def get_position_size(
    segment: MarketSegment,
    confidence: float,
    portfolio_value: float,
    is_strong_signal: bool = False,
) -> float:
    """
    Calculate position size based on segment config and confidence.
    Uses a simplified Kelly-inspired approach.
    """
    config = get_segment_config(segment)
    
    # Base position from config
    base_pct = config.base_position_pct
    
    # Scale by confidence (Kelly-lite)
    # Higher confidence = larger position
    confidence_multiplier = 0.5 + confidence  # 0.5 to 1.5
    
    # Strong signal bonus
    if is_strong_signal and confidence >= config.strong_signal_threshold:
        confidence_multiplier *= 1.2
    
    position_pct = base_pct * confidence_multiplier
    
    # Cap at max
    position_pct = min(position_pct, config.max_position_pct)
    
    position_value = portfolio_value * (position_pct / 100)
    
    return position_value


def get_stop_take_levels(
    segment: MarketSegment,
    entry_price: float,
    is_buy: bool,
) -> Tuple[float, float]:
    """
    Get stop loss and take profit levels for a segment.
    """
    config = get_segment_config(segment)
    
    sl_pct = config.stop_loss_pct / 100
    tp_pct = config.take_profit_pct / 100
    
    if is_buy:
        stop_loss = entry_price * (1 - sl_pct)
        take_profit = entry_price * (1 + tp_pct)
    else:
        stop_loss = entry_price * (1 + sl_pct)
        take_profit = entry_price * (1 - tp_pct)
    
    return stop_loss, take_profit


def get_segment_summary() -> str:
    """Get summary of segment configurations"""
    lines = ['=== SEGMENT STRATEGY CONFIGS ===', '']
    
    for segment in MarketSegment:
        config = SEGMENT_CONFIGS[segment]
        lines.append(f'{segment.value.upper()}:')
        lines.append(f'  Min Conf: {config.min_confidence:.0%} | SL: {config.stop_loss_pct}% | TP: {config.take_profit_pct}%')
        lines.append(f'  Position: {config.base_position_pct}-{config.max_position_pct}% | Contrarian: {config.contrarian_enabled}')
        lines.append('')
    
    return '\n'.join(lines)


if __name__ == '__main__':
    # Test segment detection
    test_events = [
        "Bitcoin price above $100k by March?",
        "Manchester United vs Liverpool",
        "Will Trump win 2024 election?",
        "Temperature in NYC above 80F?",
        "Team Liquid vs Cloud9 (BO3)",
        "Oscars Best Picture winner",
        "SpaceX Starship launch success?",
        "Random market question",
    ]
    
    print("Segment Detection Test:")
    print("-" * 50)
    for event in test_events:
        segment, config = get_config_for_event(event)
        print(f"{event[:40]:<40} -> {segment.value}")
    
    print("\n" + get_segment_summary())

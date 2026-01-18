"""
Market Regime Detector - Adapt to market conditions

Markets go through different phases:
- HIGH_VOLATILITY: Large swings, reduce size, wider stops
- LOW_VOLATILITY: Quiet, can be more aggressive
- TRENDING_UP: Follow momentum, don't fade
- TRENDING_DOWN: Follow momentum, don't catch falling knife
- RANGING: Mean reversion works, fade extremes
- PRE_EVENT: Close to resolution, reduce new entries
- POST_MOVE: Big move just happened, fade or wait

The system detects these regimes and adapts strategy accordingly.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from statistics import mean, stdev

logger = logging.getLogger('polymania.regime')


class MarketRegime(Enum):
    HIGH_VOLATILITY = 'high_volatility'
    LOW_VOLATILITY = 'low_volatility'
    TRENDING_UP = 'trending_up'
    TRENDING_DOWN = 'trending_down'
    RANGING = 'ranging'
    PRE_EVENT = 'pre_event'
    POST_MOVE = 'post_move'
    NORMAL = 'normal'


@dataclass
class RegimeAnalysis:
    """Analysis of current market regime"""
    regime: MarketRegime
    confidence: float  # How confident we are in this regime (0-1)
    volatility_score: float  # 0=calm, 1=wild
    trend_score: float  # -1=bearish, 0=neutral, 1=bullish
    momentum_score: float  # Recent price direction
    time_to_resolution_hours: Optional[float]
    recommendations: List[str]
    
    # Strategy adjustments
    position_size_multiplier: float = 1.0  # Multiply base position by this
    stop_loss_multiplier: float = 1.0  # Multiply stop loss by this
    confidence_adjustment: float = 0.0  # Add/subtract from signal confidence


@dataclass
class RegimeConfig:
    """Configuration for regime-based adjustments"""
    regime: MarketRegime
    position_size_multiplier: float
    stop_loss_multiplier: float
    confidence_adjustment: float
    allow_new_entries: bool
    favor_direction: Optional[str]  # 'BUY', 'SELL', or None
    description: str


# Regime-specific configurations
REGIME_CONFIGS: Dict[MarketRegime, RegimeConfig] = {
    MarketRegime.HIGH_VOLATILITY: RegimeConfig(
        regime=MarketRegime.HIGH_VOLATILITY,
        position_size_multiplier=0.5,  # Half size in volatile markets
        stop_loss_multiplier=1.5,  # Wider stops
        confidence_adjustment=-0.1,  # Be more conservative
        allow_new_entries=True,
        favor_direction=None,
        description='High volatility - reduce size, wider stops',
    ),
    
    MarketRegime.LOW_VOLATILITY: RegimeConfig(
        regime=MarketRegime.LOW_VOLATILITY,
        position_size_multiplier=1.2,  # Can be slightly more aggressive
        stop_loss_multiplier=0.8,  # Tighter stops OK
        confidence_adjustment=0.05,
        allow_new_entries=True,
        favor_direction=None,
        description='Low volatility - normal trading, tighter stops',
    ),
    
    MarketRegime.TRENDING_UP: RegimeConfig(
        regime=MarketRegime.TRENDING_UP,
        position_size_multiplier=1.1,
        stop_loss_multiplier=1.0,
        confidence_adjustment=0.1,  # Boost BUY confidence
        allow_new_entries=True,
        favor_direction='BUY',
        description='Trending up - favor BUY signals',
    ),
    
    MarketRegime.TRENDING_DOWN: RegimeConfig(
        regime=MarketRegime.TRENDING_DOWN,
        position_size_multiplier=1.1,
        stop_loss_multiplier=1.0,
        confidence_adjustment=0.1,  # Boost SELL confidence
        allow_new_entries=True,
        favor_direction='SELL',
        description='Trending down - favor SELL signals',
    ),
    
    MarketRegime.RANGING: RegimeConfig(
        regime=MarketRegime.RANGING,
        position_size_multiplier=0.9,
        stop_loss_multiplier=0.9,
        confidence_adjustment=0.0,
        allow_new_entries=True,
        favor_direction=None,
        description='Ranging - mean reversion, fade extremes',
    ),
    
    MarketRegime.PRE_EVENT: RegimeConfig(
        regime=MarketRegime.PRE_EVENT,
        position_size_multiplier=0.3,  # Very small positions
        stop_loss_multiplier=2.0,  # Wide stops if you do trade
        confidence_adjustment=-0.2,  # Much more conservative
        allow_new_entries=False,  # Avoid new entries
        favor_direction=None,
        description='Pre-event - avoid new positions, event imminent',
    ),
    
    MarketRegime.POST_MOVE: RegimeConfig(
        regime=MarketRegime.POST_MOVE,
        position_size_multiplier=0.7,
        stop_loss_multiplier=1.2,
        confidence_adjustment=-0.05,
        allow_new_entries=True,
        favor_direction=None,  # Could fade, but risky
        description='Post-move - be cautious, move may continue or reverse',
    ),
    
    MarketRegime.NORMAL: RegimeConfig(
        regime=MarketRegime.NORMAL,
        position_size_multiplier=1.0,
        stop_loss_multiplier=1.0,
        confidence_adjustment=0.0,
        allow_new_entries=True,
        favor_direction=None,
        description='Normal conditions - standard trading',
    ),
}


def calculate_volatility(prices: List[float], window: int = 20) -> float:
    """
    Calculate volatility score from price series.
    Returns 0-1 where 0 is calm and 1 is volatile.
    """
    if len(prices) < 3:
        return 0.5
    
    # Calculate returns
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(abs(ret))
    
    if not returns:
        return 0.5
    
    # Average absolute return
    avg_return = mean(returns)
    
    # Normalize to 0-1 scale
    # Typical prediction market daily moves: 0-5%
    # High volatility: >3%
    volatility_score = min(1.0, avg_return / 0.05)
    
    return volatility_score


def calculate_trend(prices: List[float], window: int = 20) -> float:
    """
    Calculate trend score from price series.
    Returns -1 (bearish) to +1 (bullish).
    """
    if len(prices) < 5:
        return 0.0
    
    recent = prices[-min(window, len(prices)):]
    
    # Simple: compare start to end
    if recent[0] == 0:
        return 0.0
    
    change = (recent[-1] - recent[0]) / recent[0]
    
    # Also check consistency (are most moves in same direction?)
    up_moves = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
    down_moves = len(recent) - 1 - up_moves
    
    if up_moves + down_moves == 0:
        consistency = 0
    else:
        consistency = (up_moves - down_moves) / (up_moves + down_moves)
    
    # Combine direction and consistency
    trend_score = (change * 5 + consistency) / 2  # Scale change by 5x
    
    # Clamp to -1 to 1
    return max(-1.0, min(1.0, trend_score))


def calculate_momentum(prices: List[float], window: int = 5) -> float:
    """
    Calculate short-term momentum.
    Returns -1 to +1.
    """
    if len(prices) < 3:
        return 0.0
    
    recent = prices[-min(window, len(prices)):]
    
    if recent[0] == 0:
        return 0.0
    
    momentum = (recent[-1] - recent[0]) / recent[0]
    
    # Scale to -1 to 1
    return max(-1.0, min(1.0, momentum * 10))


def detect_post_move(prices: List[float], threshold: float = 0.1) -> bool:
    """
    Detect if a big move just happened.
    """
    if len(prices) < 5:
        return False
    
    # Check last 5 prices for a >10% move
    recent = prices[-5:]
    
    for i in range(1, len(recent)):
        if recent[i-1] > 0:
            change = abs(recent[i] - recent[i-1]) / recent[i-1]
            if change >= threshold:
                return True
    
    return False


def estimate_time_to_resolution(event_title: str) -> Optional[float]:
    """
    Estimate hours until event resolution based on title.
    Returns None if unknown.
    """
    title_lower = event_title.lower()
    
    # Look for date patterns
    now = datetime.now(timezone.utc)
    
    # Common patterns
    if 'january' in title_lower:
        # Extract day if present
        import re
        match = re.search(r'january\s+(\d+)', title_lower)
        if match:
            day = int(match.group(1))
            target = datetime(now.year, 1, day, 23, 59, tzinfo=timezone.utc)
            if target < now:
                target = datetime(now.year + 1, 1, day, 23, 59, tzinfo=timezone.utc)
            hours = (target - now).total_seconds() / 3600
            return max(0, hours)
    
    # Sports games usually resolve within hours
    if any(x in title_lower for x in ['vs.', 'vs ', 'match', 'game']):
        return 4.0  # Assume ~4 hours
    
    # Weather is usually daily
    if any(x in title_lower for x in ['temperature', 'weather']):
        return 12.0  # Assume resolves today
    
    return None


def analyze_market_regime(
    prices: List[float],
    event_title: str = '',
    rsi: Optional[float] = None,
) -> RegimeAnalysis:
    """
    Analyze current market regime from price data and context.
    """
    recommendations = []
    
    # Calculate metrics
    volatility = calculate_volatility(prices)
    trend = calculate_trend(prices)
    momentum = calculate_momentum(prices)
    is_post_move = detect_post_move(prices)
    time_to_resolution = estimate_time_to_resolution(event_title)
    
    # Determine primary regime
    regime = MarketRegime.NORMAL
    confidence = 0.5
    
    # Check pre-event first (highest priority)
    if time_to_resolution is not None and time_to_resolution < 2.0:
        regime = MarketRegime.PRE_EVENT
        confidence = 0.9
        recommendations.append(f'Event resolving in ~{time_to_resolution:.1f}h - avoid new entries')
    
    # Check post-move
    elif is_post_move:
        regime = MarketRegime.POST_MOVE
        confidence = 0.8
        recommendations.append('Large move just occurred - be cautious')
    
    # Check high volatility
    elif volatility > 0.7:
        regime = MarketRegime.HIGH_VOLATILITY
        confidence = 0.7 + volatility * 0.2
        recommendations.append('High volatility - reduce position size')
        recommendations.append('Use wider stop losses')
    
    # Check low volatility
    elif volatility < 0.2:
        regime = MarketRegime.LOW_VOLATILITY
        confidence = 0.6
        recommendations.append('Low volatility - can use tighter stops')
    
    # Check strong trends
    elif abs(trend) > 0.6:
        if trend > 0:
            regime = MarketRegime.TRENDING_UP
            recommendations.append('Uptrend detected - favor BUY signals')
        else:
            regime = MarketRegime.TRENDING_DOWN
            recommendations.append('Downtrend detected - favor SELL signals')
        confidence = 0.5 + abs(trend) * 0.3
    
    # Check ranging
    elif abs(trend) < 0.2 and volatility > 0.3:
        regime = MarketRegime.RANGING
        confidence = 0.6
        recommendations.append('Ranging market - consider mean reversion')
    
    # Get config for adjustments
    config = REGIME_CONFIGS[regime]
    
    return RegimeAnalysis(
        regime=regime,
        confidence=confidence,
        volatility_score=volatility,
        trend_score=trend,
        momentum_score=momentum,
        time_to_resolution_hours=time_to_resolution,
        recommendations=recommendations,
        position_size_multiplier=config.position_size_multiplier,
        stop_loss_multiplier=config.stop_loss_multiplier,
        confidence_adjustment=config.confidence_adjustment,
    )


def adjust_for_regime(
    signal_type: str,
    signal_confidence: float,
    regime_analysis: RegimeAnalysis,
) -> Tuple[float, bool, List[str]]:
    """
    Adjust signal based on market regime.
    
    Returns:
        (adjusted_confidence, should_trade, reasons)
    """
    config = REGIME_CONFIGS[regime_analysis.regime]
    reasons = [f'Regime: {regime_analysis.regime.value}']
    
    adjusted = signal_confidence + regime_analysis.confidence_adjustment
    should_trade = config.allow_new_entries
    
    # Apply directional bias
    if config.favor_direction:
        is_favored = config.favor_direction in signal_type.upper()
        if is_favored:
            adjusted += 0.05
            reasons.append(f'{config.favor_direction} favored in this regime')
        else:
            adjusted -= 0.05
            reasons.append(f'{signal_type} not favored in this regime')
    
    # Add regime-specific reasons
    reasons.extend(regime_analysis.recommendations[:2])
    
    # Clamp confidence
    adjusted = max(0.1, min(0.95, adjusted))
    
    return adjusted, should_trade, reasons


def get_regime_summary() -> str:
    """Get summary of regime configurations"""
    lines = ['=== MARKET REGIME CONFIGS ===', '']
    
    for regime, config in REGIME_CONFIGS.items():
        lines.append(f'{regime.value.upper()}:')
        lines.append(f'  {config.description}')
        lines.append(f'  Size: x{config.position_size_multiplier:.1f} | SL: x{config.stop_loss_multiplier:.1f} | Conf: {config.confidence_adjustment:+.0%}')
        lines.append(f'  New Entries: {config.allow_new_entries} | Favor: {config.favor_direction or "None"}')
        lines.append('')
    
    return '\n'.join(lines)


if __name__ == '__main__':
    # Test regime detection
    print("Market Regime Detection Test")
    print("=" * 50)
    
    # Simulate different market conditions
    test_cases = [
        ("Stable market", [0.50, 0.51, 0.50, 0.52, 0.51, 0.50, 0.51]),
        ("Uptrend", [0.40, 0.42, 0.45, 0.48, 0.52, 0.55, 0.58]),
        ("Downtrend", [0.60, 0.58, 0.55, 0.52, 0.48, 0.45, 0.42]),
        ("High volatility", [0.50, 0.60, 0.45, 0.65, 0.40, 0.55, 0.35]),
        ("Post big move", [0.50, 0.50, 0.51, 0.52, 0.70, 0.72]),
    ]
    
    for name, prices in test_cases:
        analysis = analyze_market_regime(prices, "Test event")
        print(f"\n{name}:")
        print(f"  Regime: {analysis.regime.value}")
        print(f"  Volatility: {analysis.volatility_score:.2f}")
        print(f"  Trend: {analysis.trend_score:.2f}")
        print(f"  Recommendations: {', '.join(analysis.recommendations[:2])}")
    
    print("\n" + get_regime_summary())

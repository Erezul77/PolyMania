"""
Contrarian Logic - Beat the Bot Crowd

When everyone's buying (high RSI, crowded trade), that's often when the move is OVER.
This module adds skepticism to prevent us from being the last buyer in a crowded trade.

Key Principles:
1. Extreme RSI = crowded trade = reduce confidence
2. Everyone buying = we should be cautious, not aggressive
3. Reversals happen at extremes - don't chase
4. Fresh signals > stale momentum
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

logger = logging.getLogger('polymania.contrarian')


class CrowdState(Enum):
    """How crowded is this trade?"""
    EXTREME_CROWD_BUY = 'EXTREME_CROWD_BUY'   # RSI > 90, everyone's already in
    CROWDED_BUY = 'CROWDED_BUY'               # RSI > 75, getting crowded
    NORMAL = 'NORMAL'                          # RSI 30-70, healthy
    CROWDED_SELL = 'CROWDED_SELL'             # RSI < 25, panic selling
    EXTREME_CROWD_SELL = 'EXTREME_CROWD_SELL' # RSI < 10, capitulation


@dataclass
class ContrarianAnalysis:
    """Analysis of crowd behavior and our contrarian stance"""
    crowd_state: CrowdState
    confidence_penalty: float  # Reduction in confidence (0 to -0.4)
    reversal_probability: float  # Likelihood of reversal (0 to 1)
    recommendation: str
    reasons: List[str]


def analyze_crowd_state(
    rsi: Optional[float],
    price_change_pct: Optional[float],
    signal_type: str,
    consecutive_same_direction: int = 0,
) -> ContrarianAnalysis:
    """
    Analyze whether this is a crowded trade and recommend adjustments.
    
    Args:
        rsi: Current RSI value (0-100)
        price_change_pct: Recent price change percentage
        signal_type: The signal we're considering ('BUY', 'SELL', etc.)
        consecutive_same_direction: How many consecutive signals in same direction
    
    Returns:
        ContrarianAnalysis with recommendations
    """
    reasons = []
    confidence_penalty = 0.0
    reversal_probability = 0.0
    recommendation = 'PROCEED'
    
    is_buy = 'BUY' in signal_type.upper()
    is_sell = 'SELL' in signal_type.upper()
    
    # Determine crowd state from RSI
    if rsi is None:
        crowd_state = CrowdState.NORMAL
        reasons.append('No RSI data - assuming normal conditions')
    elif rsi >= 90:
        crowd_state = CrowdState.EXTREME_CROWD_BUY
        reasons.append(f'RSI {rsi:.0f} - EXTREME overbought, everyone already bought')
    elif rsi >= 75:
        crowd_state = CrowdState.CROWDED_BUY
        reasons.append(f'RSI {rsi:.0f} - Overbought, trade getting crowded')
    elif rsi <= 10:
        crowd_state = CrowdState.EXTREME_CROWD_SELL
        reasons.append(f'RSI {rsi:.0f} - EXTREME oversold, panic capitulation')
    elif rsi <= 25:
        crowd_state = CrowdState.CROWDED_SELL
        reasons.append(f'RSI {rsi:.0f} - Oversold, heavy selling')
    else:
        crowd_state = CrowdState.NORMAL
        reasons.append(f'RSI {rsi:.0f} - Normal range')
    
    # Calculate penalties and reversal probability
    if crowd_state == CrowdState.EXTREME_CROWD_BUY:
        if is_buy:
            # Trying to buy when everyone already bought - BAD IDEA
            confidence_penalty = -0.35
            reversal_probability = 0.7
            recommendation = 'SKIP - Last buyer risk'
            reasons.append('WARNING: You would be buying at extreme - likely the top')
        elif is_sell:
            # Selling when overbought - GOOD contrarian move
            confidence_penalty = 0.1  # Actually boost!
            reversal_probability = 0.7
            recommendation = 'CONTRARIAN SELL - Good timing'
            reasons.append('Contrarian opportunity: selling at extreme overbought')
    
    elif crowd_state == CrowdState.CROWDED_BUY:
        if is_buy:
            confidence_penalty = -0.2
            reversal_probability = 0.4
            recommendation = 'CAUTION - Crowded long'
            reasons.append('Trade is getting crowded - reduce size')
        elif is_sell:
            confidence_penalty = 0.05
            reversal_probability = 0.4
            recommendation = 'PROCEED - Good sell timing'
    
    elif crowd_state == CrowdState.EXTREME_CROWD_SELL:
        if is_sell:
            # Trying to sell when everyone already sold - BAD IDEA
            confidence_penalty = -0.35
            reversal_probability = 0.7
            recommendation = 'SKIP - Last seller risk'
            reasons.append('WARNING: You would be selling at extreme - likely the bottom')
        elif is_buy:
            # Buying when oversold - GOOD contrarian move
            confidence_penalty = 0.1  # Actually boost!
            reversal_probability = 0.7
            recommendation = 'CONTRARIAN BUY - Good timing'
            reasons.append('Contrarian opportunity: buying at extreme oversold')
    
    elif crowd_state == CrowdState.CROWDED_SELL:
        if is_sell:
            confidence_penalty = -0.2
            reversal_probability = 0.4
            recommendation = 'CAUTION - Crowded short'
            reasons.append('Selling pressure heavy - reduce size')
        elif is_buy:
            confidence_penalty = 0.05
            reversal_probability = 0.4
            recommendation = 'PROCEED - Good buy timing'
    
    # Additional penalty for consecutive same-direction signals
    # This indicates we might be late to the party
    if consecutive_same_direction >= 3:
        confidence_penalty -= 0.1
        reasons.append(f'{consecutive_same_direction} consecutive signals same direction - late entry risk')
    
    # Additional analysis from price change
    if price_change_pct is not None:
        if abs(price_change_pct) > 20:
            # Big move already happened
            confidence_penalty -= 0.15
            reversal_probability = min(0.9, reversal_probability + 0.2)
            reasons.append(f'Price already moved {price_change_pct:+.1f}% - most of move may be over')
        elif abs(price_change_pct) > 10:
            confidence_penalty -= 0.08
            reasons.append(f'Significant price movement ({price_change_pct:+.1f}%) already occurred')
    
    # Cap the penalty
    confidence_penalty = max(-0.4, min(0.15, confidence_penalty))
    
    return ContrarianAnalysis(
        crowd_state=crowd_state,
        confidence_penalty=confidence_penalty,
        reversal_probability=reversal_probability,
        recommendation=recommendation,
        reasons=reasons,
    )


def should_skip_trade(analysis: ContrarianAnalysis, min_confidence: float = 0.5) -> Tuple[bool, str]:
    """
    Decide if we should skip this trade based on contrarian analysis.
    
    Returns:
        (should_skip, reason)
    """
    if analysis.crowd_state in [CrowdState.EXTREME_CROWD_BUY, CrowdState.EXTREME_CROWD_SELL]:
        if analysis.confidence_penalty <= -0.3:
            return True, f'Skipping: {analysis.recommendation}'
    
    if analysis.reversal_probability >= 0.7 and analysis.confidence_penalty < 0:
        return True, f'High reversal risk ({analysis.reversal_probability:.0%})'
    
    return False, ''


def get_position_size_multiplier(analysis: ContrarianAnalysis) -> float:
    """
    Get position size multiplier based on crowd analysis.
    
    Returns multiplier from 0.25 to 1.5
    """
    if analysis.crowd_state == CrowdState.NORMAL:
        return 1.0
    
    if analysis.confidence_penalty > 0:
        # Contrarian opportunity - can size up slightly
        return min(1.3, 1.0 + analysis.confidence_penalty)
    
    if analysis.crowd_state in [CrowdState.EXTREME_CROWD_BUY, CrowdState.EXTREME_CROWD_SELL]:
        return 0.25  # Minimal size if going with crowd at extreme
    
    if analysis.crowd_state in [CrowdState.CROWDED_BUY, CrowdState.CROWDED_SELL]:
        return 0.5  # Half size in crowded trades
    
    return 1.0


def format_contrarian_warning(analysis: ContrarianAnalysis) -> Optional[str]:
    """Format a warning message for Telegram if trade is risky"""
    if analysis.confidence_penalty >= -0.1:
        return None
    
    lines = [
        '⚠️ CONTRARIAN WARNING',
        f'Crowd State: {analysis.crowd_state.value}',
        f'Recommendation: {analysis.recommendation}',
        f'Reversal Risk: {analysis.reversal_probability:.0%}',
        '',
    ]
    
    for reason in analysis.reasons[:3]:
        lines.append(f'• {reason}')
    
    return '\n'.join(lines)

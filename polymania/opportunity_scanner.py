"""
Opportunity Scanner - Advanced Pattern Recognition

Continuously scans for high-probability trading opportunities:

1. MOMENTUM PATTERNS
   - Momentum ignition (sudden acceleration)
   - Momentum exhaustion (reversal coming)
   - Momentum divergence (price vs indicators)

2. MISPRICING DETECTION
   - Odds vs implied probability gaps
   - Cross-market arbitrage opportunities
   - Event outcome mispricings

3. BREAKOUT DETECTION
   - Volume-confirmed breakouts
   - False breakout identification
   - Breakout strength scoring

4. REVERSAL PATTERNS
   - Double tops/bottoms
   - Exhaustion signals
   - Support/resistance bounces

5. TIME-BASED PATTERNS
   - Hour-of-day effects
   - Day-of-week effects
   - Pre-event patterns
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
from statistics import mean, stdev

logger = logging.getLogger('polymania.scanner')


class PatternType(Enum):
    # Momentum patterns
    MOMENTUM_IGNITION = 'momentum_ignition'
    MOMENTUM_EXHAUSTION = 'momentum_exhaustion'
    MOMENTUM_DIVERGENCE = 'momentum_divergence'
    
    # Breakout patterns
    BREAKOUT_UP = 'breakout_up'
    BREAKDOWN = 'breakdown'
    FALSE_BREAKOUT = 'false_breakout'
    
    # Reversal patterns
    DOUBLE_TOP = 'double_top'
    DOUBLE_BOTTOM = 'double_bottom'
    V_REVERSAL = 'v_reversal'
    
    # Mispricing
    MISPRICING_HIGH = 'mispricing_high'  # Price too high vs reality
    MISPRICING_LOW = 'mispricing_low'    # Price too low vs reality
    
    # Time-based
    TIME_ANOMALY = 'time_anomaly'
    PRE_EVENT_SQUEEZE = 'pre_event_squeeze'


@dataclass
class DetectedPattern:
    """A detected trading pattern"""
    pattern_type: PatternType
    strength: float  # 0 to 1
    direction: str  # 'BULLISH', 'BEARISH', or 'NEUTRAL'
    expected_move: float  # Expected % move
    time_horizon: str  # 'minutes', 'hours', 'days'
    confidence: float
    trigger_price: float
    stop_loss: float
    take_profit: float
    reasons: List[str]


@dataclass
class OpportunityScan:
    """Result of an opportunity scan"""
    event_id: str
    event_title: str
    scan_time: datetime
    patterns: List[DetectedPattern]
    overall_score: float  # -1 to +1
    recommendation: str  # 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
    urgency: str  # 'immediate', 'soon', 'watch'


class OpportunityScanner:
    """
    Advanced pattern recognition and opportunity detection.
    """
    
    def __init__(self):
        # Pattern history for learning
        self.pattern_outcomes: Dict[PatternType, Dict] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0}
        )
        
        # Time-based performance
        self.hour_performance: Dict[int, Dict] = defaultdict(
            lambda: {'wins': 0, 'losses': 0}
        )
        
        # Recent scans for trend analysis
        self.recent_scans: List[OpportunityScan] = []
        
    # =========================================================================
    # MOMENTUM PATTERNS
    # =========================================================================
    
    def detect_momentum_ignition(
        self,
        prices: List[float],
        volumes: Optional[List[float]] = None,
    ) -> Optional[DetectedPattern]:
        """
        Detect sudden momentum acceleration.
        Signs: Sharp price move with increasing velocity.
        """
        if len(prices) < 10:
            return None
        
        # Calculate recent price changes
        changes = []
        for i in range(1, min(10, len(prices))):
            if prices[i-1] != 0:
                changes.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not changes:
            return None
        
        # Check for acceleration (each move bigger than last)
        is_accelerating = all(
            abs(changes[i]) > abs(changes[i+1]) * 0.9 
            for i in range(min(3, len(changes)-1))
        )
        
        # Check magnitude
        total_move = sum(changes[:5]) if len(changes) >= 5 else sum(changes)
        
        if abs(total_move) > 0.05 and is_accelerating:  # >5% move with acceleration
            direction = 'BULLISH' if total_move > 0 else 'BEARISH'
            
            return DetectedPattern(
                pattern_type=PatternType.MOMENTUM_IGNITION,
                strength=min(1, abs(total_move) * 5),
                direction=direction,
                expected_move=total_move * 0.5,  # Expect 50% continuation
                time_horizon='hours',
                confidence=0.65,
                trigger_price=prices[-1],
                stop_loss=prices[-1] * (0.95 if direction == 'BULLISH' else 1.05),
                take_profit=prices[-1] * (1.10 if direction == 'BULLISH' else 0.90),
                reasons=['Momentum ignition detected', f'{total_move:.0%} accelerating move'],
            )
        
        return None
    
    def detect_momentum_exhaustion(
        self,
        prices: List[float],
        rsi: Optional[float] = None,
    ) -> Optional[DetectedPattern]:
        """
        Detect momentum exhaustion (potential reversal).
        Signs: Price at extreme with declining momentum.
        """
        if len(prices) < 15:
            return None
        
        # Calculate momentum decay
        recent_changes = []
        for i in range(len(prices)-5, len(prices)):
            if prices[i-1] != 0:
                recent_changes.append((prices[i] - prices[i-1]) / prices[i-1])
        
        earlier_changes = []
        for i in range(len(prices)-10, len(prices)-5):
            if prices[i-1] != 0:
                earlier_changes.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not recent_changes or not earlier_changes:
            return None
        
        recent_momentum = sum(recent_changes)
        earlier_momentum = sum(earlier_changes)
        
        # Exhaustion: momentum slowing but price still extended
        is_exhausting = (
            abs(recent_momentum) < abs(earlier_momentum) * 0.5 and
            abs(earlier_momentum) > 0.03  # Was significant momentum
        )
        
        # RSI confirmation
        rsi_confirms = False
        if rsi is not None:
            rsi_confirms = rsi > 75 or rsi < 25
        
        if is_exhausting or (rsi_confirms and abs(recent_momentum) < abs(earlier_momentum)):
            # Direction of expected reversal
            direction = 'BEARISH' if earlier_momentum > 0 else 'BULLISH'
            
            confidence = 0.55
            if rsi_confirms:
                confidence = 0.70
            
            return DetectedPattern(
                pattern_type=PatternType.MOMENTUM_EXHAUSTION,
                strength=min(1, abs(earlier_momentum) * 10),
                direction=direction,
                expected_move=-earlier_momentum * 0.5,
                time_horizon='hours',
                confidence=confidence,
                trigger_price=prices[-1],
                stop_loss=prices[-1] * (1.05 if direction == 'BEARISH' else 0.95),
                take_profit=prices[-1] * (0.92 if direction == 'BEARISH' else 1.08),
                reasons=[
                    'Momentum exhaustion',
                    f'RSI: {rsi:.0f}' if rsi else 'Velocity decay',
                ],
            )
        
        return None
    
    def detect_momentum_divergence(
        self,
        prices: List[float],
        rsi_values: Optional[List[float]] = None,
    ) -> Optional[DetectedPattern]:
        """
        Detect divergence between price and indicators.
        Bullish divergence: price lower low, RSI higher low.
        Bearish divergence: price higher high, RSI lower high.
        """
        if not rsi_values or len(prices) < 20 or len(rsi_values) < 20:
            return None
        
        # Find recent highs/lows in price
        recent_price_high = max(prices[-10:])
        recent_price_low = min(prices[-10:])
        earlier_price_high = max(prices[-20:-10])
        earlier_price_low = min(prices[-20:-10])
        
        # Find corresponding RSI
        recent_rsi_high = max(rsi_values[-10:])
        recent_rsi_low = min(rsi_values[-10:])
        earlier_rsi_high = max(rsi_values[-20:-10])
        earlier_rsi_low = min(rsi_values[-20:-10])
        
        # Bullish divergence: lower price low but higher RSI low
        if recent_price_low < earlier_price_low and recent_rsi_low > earlier_rsi_low:
            return DetectedPattern(
                pattern_type=PatternType.MOMENTUM_DIVERGENCE,
                strength=0.7,
                direction='BULLISH',
                expected_move=0.08,
                time_horizon='hours',
                confidence=0.65,
                trigger_price=prices[-1],
                stop_loss=recent_price_low * 0.97,
                take_profit=prices[-1] * 1.10,
                reasons=['Bullish RSI divergence', 'Price making lower lows, RSI higher'],
            )
        
        # Bearish divergence: higher price high but lower RSI high
        if recent_price_high > earlier_price_high and recent_rsi_high < earlier_rsi_high:
            return DetectedPattern(
                pattern_type=PatternType.MOMENTUM_DIVERGENCE,
                strength=0.7,
                direction='BEARISH',
                expected_move=-0.08,
                time_horizon='hours',
                confidence=0.65,
                trigger_price=prices[-1],
                stop_loss=recent_price_high * 1.03,
                take_profit=prices[-1] * 0.90,
                reasons=['Bearish RSI divergence', 'Price making higher highs, RSI lower'],
            )
        
        return None
    
    # =========================================================================
    # BREAKOUT PATTERNS
    # =========================================================================
    
    def detect_breakout(
        self,
        prices: List[float],
        volumes: Optional[List[float]] = None,
    ) -> Optional[DetectedPattern]:
        """
        Detect breakout from consolidation range.
        """
        if len(prices) < 30:
            return None
        
        # Calculate consolidation range (last 20-30 prices, excluding last 5)
        consolidation = prices[-30:-5]
        range_high = max(consolidation)
        range_low = min(consolidation)
        range_size = range_high - range_low
        
        if range_size == 0:
            return None
        
        # Current price position
        current = prices[-1]
        
        # Check for breakout
        breakout_up = current > range_high * 1.02  # 2% above range
        breakdown = current < range_low * 0.98  # 2% below range
        
        if not breakout_up and not breakdown:
            return None
        
        # Volume confirmation
        volume_confirmed = True
        if volumes and len(volumes) >= 30:
            recent_vol = mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
            avg_vol = mean(volumes[-30:-5])
            volume_confirmed = recent_vol > avg_vol * 1.5
        
        direction = 'BULLISH' if breakout_up else 'BEARISH'
        pattern_type = PatternType.BREAKOUT_UP if breakout_up else PatternType.BREAKDOWN
        
        # Calculate strength based on range size and breakout magnitude
        breakout_size = abs(current - (range_high if breakout_up else range_low))
        strength = min(1, (breakout_size / range_size) * 2)
        
        confidence = 0.60
        if volume_confirmed:
            confidence = 0.75
        
        return DetectedPattern(
            pattern_type=pattern_type,
            strength=strength,
            direction=direction,
            expected_move=range_size * (1 if breakout_up else -1),
            time_horizon='hours',
            confidence=confidence,
            trigger_price=current,
            stop_loss=range_high if breakout_up else range_low,  # Back into range = failed
            take_profit=current + range_size if breakout_up else current - range_size,
            reasons=[
                f'{"Breakout" if breakout_up else "Breakdown"} detected',
                'Volume confirmed' if volume_confirmed else 'Low volume - caution',
            ],
        )
    
    # =========================================================================
    # REVERSAL PATTERNS
    # =========================================================================
    
    def detect_double_pattern(
        self,
        prices: List[float],
    ) -> Optional[DetectedPattern]:
        """
        Detect double top or double bottom patterns.
        """
        if len(prices) < 40:
            return None
        
        # Find local extremes
        def find_peaks(data: List[float], is_high: bool) -> List[Tuple[int, float]]:
            peaks = []
            for i in range(2, len(data) - 2):
                if is_high:
                    if data[i] > data[i-1] and data[i] > data[i+1] and \
                       data[i] > data[i-2] and data[i] > data[i+2]:
                        peaks.append((i, data[i]))
                else:
                    if data[i] < data[i-1] and data[i] < data[i+1] and \
                       data[i] < data[i-2] and data[i] < data[i+2]:
                        peaks.append((i, data[i]))
            return peaks
        
        highs = find_peaks(prices, True)
        lows = find_peaks(prices, False)
        
        # Check for double top
        if len(highs) >= 2:
            last_two = highs[-2:]
            if len(last_two) == 2:
                h1_idx, h1_val = last_two[0]
                h2_idx, h2_val = last_two[1]
                
                # Peaks within 3% of each other and separated
                if abs(h1_val - h2_val) / h1_val < 0.03 and h2_idx - h1_idx > 5:
                    # Current price below neckline
                    neckline = min(prices[h1_idx:h2_idx])
                    if prices[-1] < neckline * 0.98:
                        return DetectedPattern(
                            pattern_type=PatternType.DOUBLE_TOP,
                            strength=0.75,
                            direction='BEARISH',
                            expected_move=-(h1_val - neckline) / neckline,
                            time_horizon='hours',
                            confidence=0.70,
                            trigger_price=prices[-1],
                            stop_loss=h1_val * 1.02,
                            take_profit=neckline - (h1_val - neckline),
                            reasons=['Double top formed', 'Neckline broken'],
                        )
        
        # Check for double bottom
        if len(lows) >= 2:
            last_two = lows[-2:]
            if len(last_two) == 2:
                l1_idx, l1_val = last_two[0]
                l2_idx, l2_val = last_two[1]
                
                if abs(l1_val - l2_val) / l1_val < 0.03 and l2_idx - l1_idx > 5:
                    neckline = max(prices[l1_idx:l2_idx])
                    if prices[-1] > neckline * 1.02:
                        return DetectedPattern(
                            pattern_type=PatternType.DOUBLE_BOTTOM,
                            strength=0.75,
                            direction='BULLISH',
                            expected_move=(neckline - l1_val) / neckline,
                            time_horizon='hours',
                            confidence=0.70,
                            trigger_price=prices[-1],
                            stop_loss=l1_val * 0.98,
                            take_profit=neckline + (neckline - l1_val),
                            reasons=['Double bottom formed', 'Neckline broken'],
                        )
        
        return None
    
    # =========================================================================
    # MISPRICING DETECTION
    # =========================================================================
    
    def detect_mispricing(
        self,
        current_price: float,
        fair_value_estimate: float,
        volatility: float,
    ) -> Optional[DetectedPattern]:
        """
        Detect when market price differs significantly from estimated fair value.
        """
        if fair_value_estimate == 0:
            return None
        
        gap = (current_price - fair_value_estimate) / fair_value_estimate
        
        # Significant mispricing threshold (adjusted for volatility)
        threshold = max(0.05, volatility * 2)
        
        if abs(gap) > threshold:
            is_overpriced = gap > 0
            
            return DetectedPattern(
                pattern_type=PatternType.MISPRICING_HIGH if is_overpriced else PatternType.MISPRICING_LOW,
                strength=min(1, abs(gap) / threshold),
                direction='BEARISH' if is_overpriced else 'BULLISH',
                expected_move=-gap * 0.5,  # Expect 50% correction
                time_horizon='days',
                confidence=0.60,
                trigger_price=current_price,
                stop_loss=current_price * (1.10 if is_overpriced else 0.90),
                take_profit=fair_value_estimate,
                reasons=[
                    f'{"Overpriced" if is_overpriced else "Underpriced"} by {abs(gap):.0%}',
                    f'Fair value estimate: {fair_value_estimate:.2f}',
                ],
            )
        
        return None
    
    # =========================================================================
    # TIME-BASED PATTERNS
    # =========================================================================
    
    def detect_time_anomaly(
        self,
        current_hour: int,
        segment: str,
    ) -> Optional[DetectedPattern]:
        """
        Detect time-based trading anomalies.
        """
        # Historical patterns by hour
        hourly_bias = self.hour_performance.get(current_hour, {})
        total = hourly_bias.get('wins', 0) + hourly_bias.get('losses', 0)
        
        if total < 20:
            return None  # Not enough data
        
        win_rate = hourly_bias['wins'] / total
        
        # Significant deviation from 50%
        if win_rate > 0.65:  # Strong bullish hour
            return DetectedPattern(
                pattern_type=PatternType.TIME_ANOMALY,
                strength=win_rate - 0.5,
                direction='BULLISH',
                expected_move=0.03,
                time_horizon='hours',
                confidence=min(0.75, 0.5 + (win_rate - 0.5) * 0.5),
                trigger_price=0,
                stop_loss=0,
                take_profit=0,
                reasons=[f'Hour {current_hour} has {win_rate:.0%} win rate', f'{total} samples'],
            )
        elif win_rate < 0.35:  # Strong bearish hour
            return DetectedPattern(
                pattern_type=PatternType.TIME_ANOMALY,
                strength=0.5 - win_rate,
                direction='BEARISH',
                expected_move=-0.03,
                time_horizon='hours',
                confidence=min(0.75, 0.5 + (0.5 - win_rate) * 0.5),
                trigger_price=0,
                stop_loss=0,
                take_profit=0,
                reasons=[f'Hour {current_hour} has {win_rate:.0%} win rate', f'{total} samples'],
            )
        
        return None
    
    # =========================================================================
    # MAIN SCANNING FUNCTION
    # =========================================================================
    
    def scan_event(
        self,
        event_id: str,
        event_title: str,
        prices: List[float],
        volumes: Optional[List[float]] = None,
        rsi: Optional[float] = None,
        rsi_values: Optional[List[float]] = None,
        fair_value: Optional[float] = None,
        segment: str = 'other',
    ) -> OpportunityScan:
        """
        Perform comprehensive opportunity scan on an event.
        """
        patterns = []
        
        # Run all pattern detectors
        detectors = [
            lambda: self.detect_momentum_ignition(prices, volumes),
            lambda: self.detect_momentum_exhaustion(prices, rsi),
            lambda: self.detect_momentum_divergence(prices, rsi_values),
            lambda: self.detect_breakout(prices, volumes),
            lambda: self.detect_double_pattern(prices),
            lambda: self.detect_time_anomaly(datetime.now(timezone.utc).hour, segment),
        ]
        
        # Add mispricing if fair value available
        if fair_value and prices:
            volatility = stdev(prices) / mean(prices) if len(prices) > 5 else 0.1
            detectors.append(
                lambda: self.detect_mispricing(prices[-1], fair_value, volatility)
            )
        
        for detector in detectors:
            try:
                pattern = detector()
                if pattern:
                    patterns.append(pattern)
            except Exception as e:
                logger.debug(f'Pattern detection error: {e}')
        
        # Calculate overall score
        if patterns:
            bullish_score = sum(
                p.strength * p.confidence 
                for p in patterns if p.direction == 'BULLISH'
            )
            bearish_score = sum(
                p.strength * p.confidence 
                for p in patterns if p.direction == 'BEARISH'
            )
            overall_score = (bullish_score - bearish_score) / max(1, len(patterns))
        else:
            overall_score = 0
        
        # Determine recommendation
        if overall_score > 0.4:
            recommendation = 'STRONG_BUY'
            urgency = 'immediate'
        elif overall_score > 0.2:
            recommendation = 'BUY'
            urgency = 'soon'
        elif overall_score < -0.4:
            recommendation = 'STRONG_SELL'
            urgency = 'immediate'
        elif overall_score < -0.2:
            recommendation = 'SELL'
            urgency = 'soon'
        else:
            recommendation = 'HOLD'
            urgency = 'watch'
        
        scan = OpportunityScan(
            event_id=event_id,
            event_title=event_title,
            scan_time=datetime.now(timezone.utc),
            patterns=patterns,
            overall_score=overall_score,
            recommendation=recommendation,
            urgency=urgency,
        )
        
        # Store for learning
        self.recent_scans.append(scan)
        if len(self.recent_scans) > 1000:
            self.recent_scans = self.recent_scans[-1000:]
        
        return scan
    
    def record_outcome(
        self,
        pattern_type: PatternType,
        won: bool,
        pnl: float,
        hour: int,
    ):
        """Record pattern outcome for learning."""
        if won:
            self.pattern_outcomes[pattern_type]['wins'] += 1
            self.hour_performance[hour]['wins'] += 1
        else:
            self.pattern_outcomes[pattern_type]['losses'] += 1
            self.hour_performance[hour]['losses'] += 1
        self.pattern_outcomes[pattern_type]['total_pnl'] += pnl
    
    def get_pattern_stats(self) -> Dict[str, Dict]:
        """Get performance stats for each pattern type."""
        stats = {}
        for pattern_type, perf in self.pattern_outcomes.items():
            total = perf['wins'] + perf['losses']
            if total > 0:
                stats[pattern_type.value] = {
                    'win_rate': perf['wins'] / total,
                    'total_trades': total,
                    'total_pnl': perf['total_pnl'],
                }
        return stats
    
    def get_summary(self) -> str:
        """Get scanner summary."""
        lines = ['=== OPPORTUNITY SCANNER STATUS ===', '']
        
        # Pattern stats
        lines.append('Pattern Performance:')
        stats = self.get_pattern_stats()
        if stats:
            for pattern, data in stats.items():
                lines.append(f'  {pattern}: {data["win_rate"]:.0%} ({data["total_trades"]} trades)')
        else:
            lines.append('  No pattern data yet')
        
        # Recent scans
        lines.append(f'\nRecent Scans: {len(self.recent_scans)}')
        
        # Recommendations distribution
        if self.recent_scans:
            rec_counts = defaultdict(int)
            for scan in self.recent_scans[-100:]:
                rec_counts[scan.recommendation] += 1
            lines.append('Recent Recommendations:')
            for rec, count in rec_counts.items():
                lines.append(f'  {rec}: {count}')
        
        return '\n'.join(lines)


# Singleton
_scanner: Optional[OpportunityScanner] = None

def get_opportunity_scanner() -> OpportunityScanner:
    global _scanner
    if _scanner is None:
        _scanner = OpportunityScanner()
    return _scanner

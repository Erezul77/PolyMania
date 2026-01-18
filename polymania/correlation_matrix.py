"""
Cross-Market Correlation Matrix

Tracks correlations between different markets to:
1. Detect spillover effects (crypto news affects crypto markets)
2. Find leading indicators (one market moves before another)
3. Identify hedging opportunities (negatively correlated markets)
4. Detect regime changes (correlations breaking down)

Example use cases:
- BTC price spike → boost confidence on all crypto markets
- Trump news → affects politics AND crypto markets  
- War news → affects geopolitics, oil, and potentially crypto
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from statistics import mean, stdev, correlation
import math

logger = logging.getLogger('polymania.correlation')


@dataclass
class MarketMove:
    """Record of a market movement"""
    event_id: str
    segment: str
    timestamp: datetime
    price_before: float
    price_after: float
    change_pct: float
    volume: float = 0


@dataclass
class CorrelationPair:
    """Correlation between two segments"""
    segment_a: str
    segment_b: str
    correlation: float  # -1 to +1
    lag_minutes: int  # A leads B by this many minutes (negative if B leads)
    sample_count: int
    last_updated: datetime
    is_significant: bool = False


@dataclass
class SpilloverSignal:
    """Signal from cross-market spillover"""
    source_segment: str
    source_event: str
    target_segment: str
    expected_impact: float  # -1 to +1
    confidence: float
    lag_remaining_minutes: int
    reason: str


class CorrelationMatrix:
    """
    Tracks and analyzes cross-market correlations.
    """
    
    # Known structural relationships
    KNOWN_RELATIONSHIPS = {
        # (segment_a, segment_b): expected_correlation
        ('crypto', 'crypto'): 0.8,  # Crypto markets move together
        ('politics', 'crypto'): 0.3,  # Political news affects crypto
        ('sports', 'sports'): 0.0,  # Sports mostly independent
        ('politics', 'politics'): 0.5,  # Political events correlate
        ('weather', 'weather'): 0.2,  # Weather events somewhat related
    }
    
    # Keywords that create cross-segment effects
    CROSS_SEGMENT_KEYWORDS = {
        'trump': ['politics', 'crypto'],
        'biden': ['politics'],
        'elon': ['crypto', 'tech'],
        'bitcoin': ['crypto'],
        'ethereum': ['crypto'],
        'fed': ['crypto', 'politics'],
        'rate': ['crypto'],
        'war': ['politics', 'crypto'],
        'israel': ['politics'],
        'iran': ['politics', 'crypto'],
        'russia': ['politics', 'crypto'],
        'china': ['politics', 'crypto', 'tech'],
    }
    
    def __init__(self):
        # Recent moves by segment
        self.recent_moves: Dict[str, List[MarketMove]] = defaultdict(list)
        
        # Calculated correlations
        self.correlations: Dict[Tuple[str, str], CorrelationPair] = {}
        
        # Moving averages by segment
        self.segment_averages: Dict[str, float] = {}
        
        # Spillover events
        self.pending_spillovers: List[SpilloverSignal] = []
        
        # Configuration
        self.max_move_history = 1000
        self.correlation_window_hours = 24
        self.min_samples_for_correlation = 10
        self.spillover_lag_minutes = 30
    
    def record_move(
        self,
        event_id: str,
        segment: str,
        price_before: float,
        price_after: float,
        volume: float = 0,
    ):
        """Record a market move for correlation analysis."""
        if price_before == 0:
            return
        
        change_pct = (price_after - price_before) / price_before
        
        move = MarketMove(
            event_id=event_id,
            segment=segment,
            timestamp=datetime.now(timezone.utc),
            price_before=price_before,
            price_after=price_after,
            change_pct=change_pct,
            volume=volume,
        )
        
        self.recent_moves[segment].append(move)
        
        # Prune old moves
        if len(self.recent_moves[segment]) > self.max_move_history:
            self.recent_moves[segment] = self.recent_moves[segment][-self.max_move_history:]
        
        # Check for spillover triggers
        if abs(change_pct) > 0.05:  # >5% move triggers spillover check
            self._trigger_spillover_check(segment, change_pct, event_id)
    
    def _trigger_spillover_check(self, source_segment: str, change_pct: float, event_id: str):
        """Check if this move should trigger spillover signals."""
        now = datetime.now(timezone.utc)
        
        # Get related segments based on known relationships
        for (seg_a, seg_b), expected_corr in self.KNOWN_RELATIONSHIPS.items():
            if seg_a == source_segment and seg_b != source_segment:
                if abs(expected_corr) > 0.2:  # Meaningful correlation
                    impact = change_pct * expected_corr
                    if abs(impact) > 0.01:  # >1% expected impact
                        self.pending_spillovers.append(SpilloverSignal(
                            source_segment=source_segment,
                            source_event=event_id,
                            target_segment=seg_b,
                            expected_impact=impact,
                            confidence=abs(expected_corr),
                            lag_remaining_minutes=self.spillover_lag_minutes,
                            reason=f'{source_segment} moved {change_pct:.0%}, expect {seg_b} follow',
                        ))
    
    def get_spillover_signals(self, target_segment: str) -> List[SpilloverSignal]:
        """Get pending spillover signals for a segment."""
        now = datetime.now(timezone.utc)
        signals = []
        
        remaining_spillovers = []
        for signal in self.pending_spillovers:
            if signal.target_segment == target_segment:
                if signal.lag_remaining_minutes > 0:
                    signals.append(signal)
                    # Decrement lag
                    signal.lag_remaining_minutes -= 1
                    if signal.lag_remaining_minutes > 0:
                        remaining_spillovers.append(signal)
            else:
                remaining_spillovers.append(signal)
        
        self.pending_spillovers = remaining_spillovers
        return signals
    
    def calculate_correlation(self, segment_a: str, segment_b: str) -> Optional[CorrelationPair]:
        """Calculate correlation between two segments."""
        moves_a = self.recent_moves.get(segment_a, [])
        moves_b = self.recent_moves.get(segment_b, [])
        
        if len(moves_a) < self.min_samples_for_correlation or \
           len(moves_b) < self.min_samples_for_correlation:
            return None
        
        # Filter to correlation window
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.correlation_window_hours)
        moves_a = [m for m in moves_a if m.timestamp > cutoff]
        moves_b = [m for m in moves_b if m.timestamp > cutoff]
        
        if len(moves_a) < self.min_samples_for_correlation or \
           len(moves_b) < self.min_samples_for_correlation:
            return None
        
        # Extract change percentages
        changes_a = [m.change_pct for m in moves_a]
        changes_b = [m.change_pct for m in moves_b]
        
        # Pad shorter list
        max_len = min(len(changes_a), len(changes_b))
        changes_a = changes_a[-max_len:]
        changes_b = changes_b[-max_len:]
        
        # Calculate correlation
        try:
            corr = correlation(changes_a, changes_b)
        except Exception:
            corr = 0
        
        is_significant = abs(corr) > 0.3 and max_len >= 20
        
        result = CorrelationPair(
            segment_a=segment_a,
            segment_b=segment_b,
            correlation=corr,
            lag_minutes=0,  # TODO: calculate lag
            sample_count=max_len,
            last_updated=datetime.now(timezone.utc),
            is_significant=is_significant,
        )
        
        self.correlations[(segment_a, segment_b)] = result
        return result
    
    def get_cross_market_boost(
        self,
        event_title: str,
        segment: str,
        signal_direction: str,  # 'BUY' or 'SELL'
    ) -> Tuple[float, List[str]]:
        """
        Get confidence adjustment from cross-market analysis.
        
        Returns:
            (adjustment, reasons)
        """
        adjustment = 0
        reasons = []
        
        # Check for keyword-based cross effects
        title_lower = event_title.lower()
        for keyword, affected_segments in self.CROSS_SEGMENT_KEYWORDS.items():
            if keyword in title_lower and segment in affected_segments:
                # Check if related segments have recent moves
                for other_segment in affected_segments:
                    if other_segment != segment:
                        recent = self.recent_moves.get(other_segment, [])
                        if recent:
                            # Get average recent move direction
                            cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
                            recent_changes = [
                                m.change_pct for m in recent 
                                if m.timestamp > cutoff
                            ]
                            if recent_changes:
                                avg_change = mean(recent_changes)
                                if abs(avg_change) > 0.02:
                                    # Related market moved significantly
                                    if (avg_change > 0 and signal_direction == 'BUY') or \
                                       (avg_change < 0 and signal_direction == 'SELL'):
                                        adjustment += 0.05
                                        reasons.append(f'{other_segment} confirming ({avg_change:+.0%})')
                                    else:
                                        adjustment -= 0.03
                                        reasons.append(f'{other_segment} diverging')
        
        # Check spillover signals
        spillovers = self.get_spillover_signals(segment)
        for signal in spillovers:
            if (signal.expected_impact > 0 and signal_direction == 'BUY') or \
               (signal.expected_impact < 0 and signal_direction == 'SELL'):
                adjustment += 0.05 * signal.confidence
                reasons.append(f'Spillover from {signal.source_segment}')
        
        return adjustment, reasons
    
    def get_regime_correlation_breakdown(self) -> Dict[str, float]:
        """
        Detect if correlations are breaking down (regime change indicator).
        
        Returns correlation stability by segment.
        """
        stability = {}
        
        for segment in self.recent_moves:
            if segment not in self.segment_averages:
                continue
            
            recent = self.recent_moves[segment][-50:]  # Last 50 moves
            if len(recent) < 20:
                stability[segment] = 1.0
                continue
            
            changes = [m.change_pct for m in recent]
            
            # Calculate rolling correlation with itself (autocorrelation)
            # High autocorrelation = trending, low = ranging
            if len(changes) >= 10:
                first_half = changes[:len(changes)//2]
                second_half = changes[len(changes)//2:]
                try:
                    min_len = min(len(first_half), len(second_half))
                    autocorr = correlation(first_half[:min_len], second_half[:min_len])
                    stability[segment] = abs(autocorr)
                except Exception:
                    stability[segment] = 0.5
            else:
                stability[segment] = 0.5
        
        return stability
    
    def get_summary(self) -> str:
        """Get correlation matrix summary."""
        lines = ['=== CROSS-MARKET CORRELATION MATRIX ===', '']
        
        # Segment stats
        lines.append('Move History by Segment:')
        for segment, moves in self.recent_moves.items():
            if moves:
                avg = mean(m.change_pct for m in moves[-20:]) if len(moves) >= 20 else 0
                lines.append(f'  {segment}: {len(moves)} moves, avg {avg:+.1%}')
        
        # Significant correlations
        lines.append('\nSignificant Correlations:')
        sig_corrs = [c for c in self.correlations.values() if c.is_significant]
        if sig_corrs:
            for c in sig_corrs:
                lines.append(f'  {c.segment_a} ↔ {c.segment_b}: {c.correlation:.2f}')
        else:
            lines.append('  None detected yet')
        
        # Pending spillovers
        lines.append(f'\nPending Spillovers: {len(self.pending_spillovers)}')
        for s in self.pending_spillovers[:3]:
            lines.append(f'  {s.source_segment} → {s.target_segment}: {s.expected_impact:+.0%}')
        
        return '\n'.join(lines)


# Singleton
_matrix: Optional[CorrelationMatrix] = None

def get_correlation_matrix() -> CorrelationMatrix:
    global _matrix
    if _matrix is None:
        _matrix = CorrelationMatrix()
    return _matrix

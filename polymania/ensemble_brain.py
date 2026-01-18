"""
Ensemble Brain - The Ultimate Decision Maker

This is the master orchestrator that combines ALL analysis modules:

1. SIGNAL AGGREGATION
   - Combine signals from all sources
   - Weight by historical performance
   - Detect conflicts and consensus

2. CONFIDENCE CALIBRATION
   - Adjust confidence based on agreement
   - Account for market regime
   - Learn from outcomes

3. POSITION SIZING (Kelly Criterion++)
   - Optimal bet sizing based on edge
   - Risk-adjusted for volatility
   - Drawdown protection

4. TIMING OPTIMIZATION
   - Best entry points
   - Exit timing
   - Event timing considerations

5. CONTINUOUS LEARNING
   - Track all decisions
   - Update weights based on outcomes
   - Evolve strategy over time
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict
from statistics import mean, stdev
import json
import os

logger = logging.getLogger('polymania.brain')


class SignalSource(Enum):
    TECHNICAL = 'technical'
    MOMENTUM = 'momentum'
    PATTERN = 'pattern'
    WHALE = 'whale'
    NEWS = 'news'
    TELEGRAM = 'telegram'
    CROSS_MARKET = 'cross_market'
    STRATEGY_LEARNED = 'strategy_learned'
    REGIME = 'regime'


@dataclass
class SourceSignal:
    """Signal from a specific source"""
    source: SignalSource
    direction: str  # 'BUY', 'SELL', 'NEUTRAL'
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    reasons: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EnsembleDecision:
    """Final ensemble decision"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    position_size: float  # As fraction of portfolio (0-1)
    entry_price: float
    stop_loss: float
    take_profit: float
    max_holding_time: timedelta
    sources_agreeing: int
    sources_disagreeing: int
    consensus_level: float  # 0 to 1
    edge_estimate: float
    reasons: List[str]
    signal_breakdown: Dict[str, float]


@dataclass
class PerformanceRecord:
    """Record of a decision's outcome"""
    decision: EnsembleDecision
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    source_contributions: Dict[str, float]


class EnsembleBrain:
    """
    The master decision maker that combines all signals.
    """
    
    def __init__(self):
        # Source performance tracking
        self.source_performance: Dict[SignalSource, Dict] = defaultdict(
            lambda: {
                'predictions': 0,
                'correct': 0,
                'total_pnl': 0,
                'avg_confidence_when_right': 0,
                'avg_confidence_when_wrong': 0,
            }
        )
        
        # Combined performance
        self.ensemble_performance = {
            'decisions': 0,
            'wins': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'current_drawdown': 0,
        }
        
        # Recent decisions for analysis
        self.recent_decisions: List[PerformanceRecord] = []
        
        # Adaptive weights
        self.source_weights = self._init_weights()
        
        # Risk parameters
        self.max_position_size = 0.15  # Max 15% per position
        self.max_daily_risk = 0.25  # Max 25% daily risk
        self.current_daily_risk = 0
        
        # Data persistence
        self.data_file = 'data/ensemble_brain.json'
        self._load_state()
    
    def _init_weights(self) -> Dict[SignalSource, float]:
        """Initialize source weights."""
        return {
            SignalSource.TECHNICAL: 1.0,
            SignalSource.MOMENTUM: 1.1,
            SignalSource.PATTERN: 0.9,
            SignalSource.WHALE: 1.3,
            SignalSource.NEWS: 0.8,
            SignalSource.TELEGRAM: 0.7,
            SignalSource.CROSS_MARKET: 0.6,
            SignalSource.STRATEGY_LEARNED: 1.2,
            SignalSource.REGIME: 0.8,
        }
    
    # =========================================================================
    # SIGNAL AGGREGATION
    # =========================================================================
    
    def aggregate_signals(
        self,
        signals: List[SourceSignal],
    ) -> Tuple[str, float, float, Dict[str, float]]:
        """
        Aggregate signals from all sources.
        
        Returns:
            (direction, weighted_strength, consensus, breakdown)
        """
        if not signals:
            return 'NEUTRAL', 0, 0, {}
        
        # Separate by direction
        buy_signals = [s for s in signals if s.direction == 'BUY']
        sell_signals = [s for s in signals if s.direction == 'SELL']
        
        # Calculate weighted scores
        def weighted_score(signal_list: List[SourceSignal]) -> float:
            if not signal_list:
                return 0
            total = sum(
                s.strength * s.confidence * self.source_weights.get(s.source, 1.0)
                for s in signal_list
            )
            return total
        
        buy_score = weighted_score(buy_signals)
        sell_score = weighted_score(sell_signals)
        
        # Determine direction
        if buy_score > sell_score * 1.2:  # 20% margin required
            direction = 'BUY'
            strength = buy_score / (buy_score + sell_score) if (buy_score + sell_score) > 0 else 0
        elif sell_score > buy_score * 1.2:
            direction = 'SELL'
            strength = sell_score / (buy_score + sell_score) if (buy_score + sell_score) > 0 else 0
        else:
            direction = 'NEUTRAL'
            strength = 0
        
        # Calculate consensus (how much sources agree)
        total_sources = len(signals)
        agreeing = len(buy_signals) if direction == 'BUY' else len(sell_signals)
        consensus = agreeing / total_sources if total_sources > 0 else 0
        
        # Breakdown by source
        breakdown = {}
        for signal in signals:
            source_name = signal.source.value
            contribution = signal.strength * signal.confidence * self.source_weights.get(signal.source, 1.0)
            if signal.direction != direction and direction != 'NEUTRAL':
                contribution = -contribution  # Negative for opposing signals
            breakdown[source_name] = contribution
        
        return direction, strength, consensus, breakdown
    
    # =========================================================================
    # CONFIDENCE CALIBRATION
    # =========================================================================
    
    def calibrate_confidence(
        self,
        raw_confidence: float,
        consensus: float,
        regime: str,
        signal_count: int,
    ) -> float:
        """
        Calibrate confidence based on various factors.
        """
        confidence = raw_confidence
        
        # Consensus adjustment
        if consensus > 0.8:
            confidence *= 1.15  # High agreement boost
        elif consensus < 0.5:
            confidence *= 0.8  # Low agreement penalty
        
        # Signal count adjustment (more signals = more conviction)
        if signal_count >= 5:
            confidence *= 1.1
        elif signal_count <= 2:
            confidence *= 0.85
        
        # Regime adjustment
        regime_factors = {
            'high_volatility': 0.8,
            'low_volatility': 1.1,
            'trending_up': 1.05,
            'trending_down': 1.05,
            'pre_event': 0.7,
            'post_move': 0.85,
            'normal': 1.0,
        }
        confidence *= regime_factors.get(regime, 1.0)
        
        # Historical calibration (are we overconfident?)
        if self.ensemble_performance['decisions'] > 20:
            historical_accuracy = (
                self.ensemble_performance['wins'] / 
                self.ensemble_performance['decisions']
            )
            # Adjust towards historical accuracy
            confidence = 0.7 * confidence + 0.3 * historical_accuracy
        
        return max(0.1, min(0.95, confidence))
    
    # =========================================================================
    # POSITION SIZING (Kelly Criterion++)
    # =========================================================================
    
    def calculate_position_size(
        self,
        confidence: float,
        edge_estimate: float,
        volatility: float,
        current_drawdown: float,
    ) -> float:
        """
        Calculate optimal position size using enhanced Kelly Criterion.
        
        Kelly formula: f* = (bp - q) / b
        Where: b = odds, p = win probability, q = lose probability
        
        We use half-Kelly for safety.
        """
        # Estimate odds based on market structure
        # In prediction markets, typical odds are around 1:1 to 2:1
        estimated_odds = 1.5
        
        # Win probability from confidence
        win_prob = confidence
        lose_prob = 1 - win_prob
        
        # Kelly fraction
        kelly = (estimated_odds * win_prob - lose_prob) / estimated_odds
        
        # Use half-Kelly for safety
        half_kelly = kelly / 2
        
        # Volatility adjustment (lower size in high vol)
        vol_factor = 1.0 / (1 + volatility * 2)
        
        # Drawdown protection (reduce size during drawdowns)
        drawdown_factor = 1.0
        if current_drawdown > 0.05:
            drawdown_factor = 0.7
        elif current_drawdown > 0.10:
            drawdown_factor = 0.5
        elif current_drawdown > 0.20:
            drawdown_factor = 0.25
        
        # Edge adjustment
        edge_factor = 1.0 + edge_estimate
        
        # Calculate final size
        position_size = half_kelly * vol_factor * drawdown_factor * edge_factor
        
        # Apply limits
        position_size = max(0.01, min(self.max_position_size, position_size))
        
        # Check daily risk limit
        if self.current_daily_risk + position_size > self.max_daily_risk:
            position_size = max(0, self.max_daily_risk - self.current_daily_risk)
        
        return position_size
    
    # =========================================================================
    # TIMING OPTIMIZATION
    # =========================================================================
    
    def optimize_entry(
        self,
        current_price: float,
        recent_prices: List[float],
        direction: str,
    ) -> Tuple[float, str]:
        """
        Optimize entry timing.
        
        Returns:
            (suggested_entry_price, timing_advice)
        """
        if not recent_prices or len(recent_prices) < 10:
            return current_price, 'immediate'
        
        # Calculate support/resistance
        recent_high = max(recent_prices[-20:]) if len(recent_prices) >= 20 else max(recent_prices)
        recent_low = min(recent_prices[-20:]) if len(recent_prices) >= 20 else min(recent_prices)
        range_size = recent_high - recent_low
        
        if direction == 'BUY':
            # Look for pullback entry
            ideal_entry = recent_low + range_size * 0.3  # Lower third of range
            if current_price > ideal_entry * 1.05:
                return ideal_entry, 'wait_for_pullback'
            else:
                return current_price, 'immediate'
        else:
            # Look for bounce entry
            ideal_entry = recent_high - range_size * 0.3  # Upper third
            if current_price < ideal_entry * 0.95:
                return ideal_entry, 'wait_for_bounce'
            else:
                return current_price, 'immediate'
    
    def calculate_exits(
        self,
        entry_price: float,
        direction: str,
        confidence: float,
        volatility: float,
    ) -> Tuple[float, float, timedelta]:
        """
        Calculate stop loss, take profit, and max holding time.
        """
        # Base percentages
        if direction == 'BUY':
            base_stop = 0.05  # 5% stop
            base_profit = 0.10  # 10% target
        else:
            base_stop = 0.05
            base_profit = 0.10
        
        # Adjust for volatility
        vol_factor = 1 + volatility
        stop_loss_pct = base_stop * vol_factor
        take_profit_pct = base_profit * vol_factor
        
        # Adjust for confidence (higher confidence = tighter stops, bigger targets)
        if confidence > 0.75:
            stop_loss_pct *= 0.8
            take_profit_pct *= 1.2
        elif confidence < 0.5:
            stop_loss_pct *= 1.2
            take_profit_pct *= 0.8
        
        # Calculate prices
        if direction == 'BUY':
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
        
        # Max holding time based on confidence
        if confidence > 0.7:
            max_time = timedelta(days=7)
        elif confidence > 0.5:
            max_time = timedelta(days=3)
        else:
            max_time = timedelta(days=1)
        
        return stop_loss, take_profit, max_time
    
    # =========================================================================
    # MAIN DECISION FUNCTION
    # =========================================================================
    
    def make_decision(
        self,
        event_id: str,
        event_title: str,
        current_price: float,
        recent_prices: List[float],
        signals: List[SourceSignal],
        regime: str = 'normal',
        volatility: float = 0.1,
    ) -> EnsembleDecision:
        """
        Make the final ensemble decision.
        """
        # Aggregate signals
        direction, strength, consensus, breakdown = self.aggregate_signals(signals)
        
        # Calculate edge estimate
        edge_estimate = (strength - 0.5) * consensus  # Higher when strong AND consensual
        
        # Calibrate confidence
        raw_confidence = strength * consensus
        confidence = self.calibrate_confidence(
            raw_confidence, consensus, regime, len(signals)
        )
        
        # Determine action
        if direction == 'NEUTRAL' or confidence < 0.4:
            action = 'HOLD'
        else:
            action = direction
        
        # Calculate position size
        position_size = 0
        if action != 'HOLD':
            position_size = self.calculate_position_size(
                confidence, edge_estimate, volatility, 
                self.ensemble_performance['current_drawdown']
            )
        
        # Optimize entry
        entry_price, timing = self.optimize_entry(current_price, recent_prices, direction)
        
        # Calculate exits
        stop_loss, take_profit, max_time = self.calculate_exits(
            entry_price, direction, confidence, volatility
        )
        
        # Count agreeing/disagreeing sources
        agreeing = sum(1 for s in signals if s.direction == direction)
        disagreeing = sum(1 for s in signals if s.direction != direction and s.direction != 'NEUTRAL')
        
        # Collect reasons
        reasons = []
        for signal in signals:
            if signal.direction == direction:
                reasons.extend(signal.reasons[:1])
        reasons = reasons[:5]  # Top 5 reasons
        
        if timing != 'immediate':
            reasons.append(f'Timing: {timing}')
        
        decision = EnsembleDecision(
            action=action,
            confidence=confidence,
            position_size=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_holding_time=max_time,
            sources_agreeing=agreeing,
            sources_disagreeing=disagreeing,
            consensus_level=consensus,
            edge_estimate=edge_estimate,
            reasons=reasons,
            signal_breakdown=breakdown,
        )
        
        # Track decision
        self.ensemble_performance['decisions'] += 1
        
        return decision
    
    # =========================================================================
    # LEARNING & ADAPTATION
    # =========================================================================
    
    def record_outcome(
        self,
        decision: EnsembleDecision,
        entry_price: float,
        exit_price: float,
        pnl: float,
    ):
        """
        Record outcome and update weights.
        """
        won = pnl > 0
        
        # Update ensemble performance
        if won:
            self.ensemble_performance['wins'] += 1
        self.ensemble_performance['total_pnl'] += pnl
        
        # Update drawdown
        if pnl < 0:
            self.ensemble_performance['current_drawdown'] += abs(pnl)
            self.ensemble_performance['max_drawdown'] = max(
                self.ensemble_performance['max_drawdown'],
                self.ensemble_performance['current_drawdown']
            )
        else:
            self.ensemble_performance['current_drawdown'] = max(
                0, self.ensemble_performance['current_drawdown'] - pnl
            )
        
        # Update source weights based on contribution
        for source_name, contribution in decision.signal_breakdown.items():
            try:
                source = SignalSource(source_name)
                perf = self.source_performance[source]
                perf['predictions'] += 1
                
                if (contribution > 0 and won) or (contribution < 0 and not won):
                    perf['correct'] += 1
                    # Increase weight for correct sources
                    self.source_weights[source] = min(2.0, self.source_weights[source] * 1.02)
                else:
                    # Decrease weight for incorrect sources
                    self.source_weights[source] = max(0.3, self.source_weights[source] * 0.98)
                
                perf['total_pnl'] += pnl * abs(contribution) / sum(abs(c) for c in decision.signal_breakdown.values())
            except (ValueError, ZeroDivisionError):
                continue
        
        # Save state
        self._save_state()
    
    def _save_state(self):
        """Save state to disk."""
        try:
            os.makedirs('data', exist_ok=True)
            state = {
                'ensemble_performance': self.ensemble_performance,
                'source_weights': {k.value: v for k, v in self.source_weights.items()},
                'source_performance': {
                    k.value: v for k, v in self.source_performance.items()
                },
            }
            with open(self.data_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f'Failed to save brain state: {e}')
    
    def _load_state(self):
        """Load state from disk."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    state = json.load(f)
                
                self.ensemble_performance.update(state.get('ensemble_performance', {}))
                
                for k, v in state.get('source_weights', {}).items():
                    try:
                        self.source_weights[SignalSource(k)] = v
                    except ValueError:
                        pass
                
                for k, v in state.get('source_performance', {}).items():
                    try:
                        self.source_performance[SignalSource(k)].update(v)
                    except ValueError:
                        pass
        except Exception as e:
            logger.warning(f'Failed to load brain state: {e}')
    
    def get_source_rankings(self) -> List[Tuple[str, float, float]]:
        """Get sources ranked by performance."""
        rankings = []
        for source, perf in self.source_performance.items():
            if perf['predictions'] > 0:
                accuracy = perf['correct'] / perf['predictions']
                weight = self.source_weights.get(source, 1.0)
                rankings.append((source.value, accuracy, weight))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_summary(self) -> str:
        """Get brain summary."""
        lines = ['=== ENSEMBLE BRAIN STATUS ===', '']
        
        # Ensemble performance
        perf = self.ensemble_performance
        total = perf['decisions']
        if total > 0:
            win_rate = perf['wins'] / total
            lines.append(f'Decisions: {total}')
            lines.append(f'Win Rate: {win_rate:.0%}')
            lines.append(f'Total PnL: {perf["total_pnl"]:.2f}')
            lines.append(f'Max Drawdown: {perf["max_drawdown"]:.1%}')
            lines.append(f'Current Drawdown: {perf["current_drawdown"]:.1%}')
        else:
            lines.append('No decisions yet')
        
        # Source rankings
        lines.append('\nSource Rankings (by accuracy):')
        rankings = self.get_source_rankings()
        if rankings:
            for name, acc, weight in rankings[:5]:
                lines.append(f'  {name}: {acc:.0%} accuracy, {weight:.2f}x weight')
        else:
            lines.append('  No source data yet')
        
        # Current weights
        lines.append('\nCurrent Weights:')
        for source, weight in sorted(self.source_weights.items(), key=lambda x: x[1], reverse=True):
            lines.append(f'  {source.value}: {weight:.2f}')
        
        return '\n'.join(lines)


# Singleton
_brain: Optional[EnsembleBrain] = None

def get_ensemble_brain() -> EnsembleBrain:
    global _brain
    if _brain is None:
        _brain = EnsembleBrain()
    return _brain

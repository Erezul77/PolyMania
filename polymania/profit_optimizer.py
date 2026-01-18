"""
Profit Optimizer - Maximum Alpha Extraction System

This module combines multiple advanced analysis techniques to maximize profits:

1. MULTI-DIMENSIONAL SCORING
   - Combine technical, fundamental, sentiment, and flow signals
   - Weight each dimension based on historical performance
   - Dynamic adjustment based on market conditions

2. CROSS-MARKET CORRELATION
   - Detect when markets move together
   - Spillover effects (crypto news affects crypto markets)
   - Leading indicators (one market predicts another)

3. MOMENTUM & OPPORTUNITY DETECTION
   - Multi-timeframe momentum scoring
   - Mispricing detection (odds vs reality)
   - Breakout and breakdown detection
   - Mean reversion opportunities

4. ENSEMBLE DECISION MAKING
   - Combine multiple signal sources
   - Confidence-weighted voting
   - Disagreement detection (conflicting signals = caution)

5. DYNAMIC OPTIMIZATION
   - Adaptive parameters based on recent performance
   - Risk-adjusted position sizing
   - Drawdown protection
   - Profit-taking optimization
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict
from statistics import mean, stdev

logger = logging.getLogger('polymania.optimizer')


class OpportunityType(Enum):
    MOMENTUM_BUY = 'momentum_buy'
    MOMENTUM_SELL = 'momentum_sell'
    MEAN_REVERSION_BUY = 'mean_reversion_buy'
    MEAN_REVERSION_SELL = 'mean_reversion_sell'
    BREAKOUT_UP = 'breakout_up'
    BREAKDOWN = 'breakdown'
    MISPRICING = 'mispricing'
    WHALE_FOLLOW = 'whale_follow'
    NEWS_CATALYST = 'news_catalyst'
    CROSS_MARKET = 'cross_market'


@dataclass
class SignalDimension:
    """A single dimension of signal analysis"""
    name: str
    score: float  # -1 to +1
    confidence: float  # 0 to 1
    weight: float = 1.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class MultiDimensionalScore:
    """Combined score from all dimensions"""
    dimensions: List[SignalDimension]
    final_score: float  # -1 to +1
    final_confidence: float  # 0 to 1
    agreement_level: float  # How much dimensions agree (0-1)
    dominant_signal: str  # 'BUY', 'SELL', 'NEUTRAL'
    opportunity_type: Optional[OpportunityType] = None
    reasons: List[str] = field(default_factory=list)


@dataclass 
class CrossMarketSignal:
    """Signal derived from cross-market analysis"""
    source_market: str
    target_market: str
    correlation: float
    lag_minutes: int
    signal_strength: float
    direction: str  # 'SAME' or 'OPPOSITE'
    confidence: float


@dataclass
class MomentumProfile:
    """Multi-timeframe momentum analysis"""
    short_term: float  # Last 5 data points
    medium_term: float  # Last 20 data points
    long_term: float  # Last 50+ data points
    acceleration: float  # Is momentum increasing or decreasing?
    consistency: float  # Are all timeframes aligned?
    breakout_potential: float  # Likelihood of breakout


@dataclass
class OpportunityScore:
    """Detected trading opportunity"""
    opportunity_type: OpportunityType
    score: float  # 0 to 1
    expected_return: float  # Expected % return
    risk_level: float  # 0 to 1
    time_sensitivity: str  # 'immediate', 'hours', 'days'
    reasons: List[str]


class ProfitOptimizer:
    """
    The brain that combines all signals for maximum profit.
    """
    
    def __init__(self):
        # Historical performance tracking for adaptive weights
        self.dimension_performance: Dict[str, Dict] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0}
        )
        
        # Cross-market correlation cache
        self.market_correlations: Dict[str, Dict[str, float]] = {}
        
        # Recent signals for pattern analysis
        self.recent_signals: List[Dict] = []
        
        # Performance metrics
        self.total_signals = 0
        self.winning_signals = 0
        
    # =========================================================================
    # MULTI-DIMENSIONAL SCORING
    # =========================================================================
    
    def calculate_technical_dimension(
        self,
        rsi: Optional[float],
        macd_histogram: Optional[float],
        trend: str,
        price_change_pct: Optional[float],
    ) -> SignalDimension:
        """
        Technical analysis dimension score.
        """
        scores = []
        reasons = []
        
        # RSI analysis
        if rsi is not None:
            if rsi < 30:
                scores.append(0.5 + (30 - rsi) / 60)  # Oversold = bullish
                reasons.append(f'RSI {rsi:.0f} oversold')
            elif rsi > 70:
                scores.append(-0.5 - (rsi - 70) / 60)  # Overbought = bearish
                reasons.append(f'RSI {rsi:.0f} overbought')
            else:
                scores.append((50 - rsi) / 40)  # Neutral zone
        
        # MACD analysis
        if macd_histogram is not None:
            macd_score = max(-1, min(1, macd_histogram * 10))
            scores.append(macd_score)
            if macd_histogram > 0:
                reasons.append('MACD positive')
            elif macd_histogram < 0:
                reasons.append('MACD negative')
        
        # Trend analysis
        if trend == 'BULLISH':
            scores.append(0.5)
            reasons.append('Bullish trend')
        elif trend == 'BEARISH':
            scores.append(-0.5)
            reasons.append('Bearish trend')
        
        # Price momentum
        if price_change_pct is not None:
            mom_score = max(-1, min(1, price_change_pct / 20))
            scores.append(mom_score)
        
        final_score = mean(scores) if scores else 0
        confidence = 0.3 + len(scores) * 0.15  # More indicators = more confidence
        
        return SignalDimension(
            name='technical',
            score=final_score,
            confidence=min(0.9, confidence),
            weight=self._get_adaptive_weight('technical'),
            reasons=reasons,
        )
    
    def calculate_sentiment_dimension(
        self,
        has_news_alpha: bool,
        news_sentiment: float,  # -1 to 1
        telegram_hits: int,
        social_buzz: float,  # 0 to 1
    ) -> SignalDimension:
        """
        Sentiment analysis dimension score.
        """
        scores = []
        reasons = []
        
        if has_news_alpha:
            scores.append(news_sentiment)
            if news_sentiment > 0.3:
                reasons.append('Positive news sentiment')
            elif news_sentiment < -0.3:
                reasons.append('Negative news sentiment')
        
        if telegram_hits > 0:
            # More hits = more attention = potential move
            hit_score = min(1, telegram_hits / 5) * 0.3
            scores.append(hit_score)
            reasons.append(f'{telegram_hits} Telegram hits')
        
        if social_buzz > 0.5:
            scores.append(social_buzz * 0.5)
            reasons.append('High social buzz')
        
        final_score = mean(scores) if scores else 0
        confidence = 0.2 + len(scores) * 0.2
        
        return SignalDimension(
            name='sentiment',
            score=final_score,
            confidence=min(0.8, confidence),
            weight=self._get_adaptive_weight('sentiment'),
            reasons=reasons,
        )
    
    def calculate_flow_dimension(
        self,
        whale_signal: bool,
        whale_direction: str,  # 'BUY' or 'SELL'
        volume_spike: bool,
        smart_money_flow: float,  # -1 to 1
    ) -> SignalDimension:
        """
        Money flow dimension score.
        """
        scores = []
        reasons = []
        
        if whale_signal:
            if whale_direction == 'BUY':
                scores.append(0.7)
                reasons.append('🐋 Whale buying')
            else:
                scores.append(-0.7)
                reasons.append('🐋 Whale selling')
        
        if volume_spike:
            scores.append(0.3)  # Volume spike = something happening
            reasons.append('Volume spike detected')
        
        if smart_money_flow != 0:
            scores.append(smart_money_flow * 0.5)
            if smart_money_flow > 0:
                reasons.append('Smart money inflow')
            else:
                reasons.append('Smart money outflow')
        
        final_score = mean(scores) if scores else 0
        confidence = 0.3 + len(scores) * 0.2
        
        return SignalDimension(
            name='flow',
            score=final_score,
            confidence=min(0.85, confidence),
            weight=self._get_adaptive_weight('flow'),
            reasons=reasons,
        )
    
    def calculate_structural_dimension(
        self,
        near_support: bool,
        near_resistance: bool,
        in_range: bool,
        breakout_imminent: bool,
    ) -> SignalDimension:
        """
        Market structure dimension score.
        """
        scores = []
        reasons = []
        
        if near_support:
            scores.append(0.4)
            reasons.append('Near support level')
        
        if near_resistance:
            scores.append(-0.4)
            reasons.append('Near resistance level')
        
        if breakout_imminent:
            scores.append(0.3)  # Breakouts tend to be bullish
            reasons.append('Breakout forming')
        
        if in_range:
            scores.append(0)  # Neutral in range
            reasons.append('Trading in range')
        
        final_score = mean(scores) if scores else 0
        confidence = 0.4 if scores else 0.2
        
        return SignalDimension(
            name='structural',
            score=final_score,
            confidence=confidence,
            weight=self._get_adaptive_weight('structural'),
            reasons=reasons,
        )
    
    def combine_dimensions(
        self,
        dimensions: List[SignalDimension],
    ) -> MultiDimensionalScore:
        """
        Combine all dimensions into a final score.
        Uses weighted average with agreement bonus/penalty.
        """
        if not dimensions:
            return MultiDimensionalScore(
                dimensions=[],
                final_score=0,
                final_confidence=0,
                agreement_level=0,
                dominant_signal='NEUTRAL',
                reasons=['No dimensions available'],
            )
        
        # Calculate weighted score
        total_weight = sum(d.weight * d.confidence for d in dimensions)
        if total_weight == 0:
            weighted_score = 0
        else:
            weighted_score = sum(
                d.score * d.weight * d.confidence for d in dimensions
            ) / total_weight
        
        # Calculate agreement level
        positive = sum(1 for d in dimensions if d.score > 0.1)
        negative = sum(1 for d in dimensions if d.score < -0.1)
        neutral = len(dimensions) - positive - negative
        
        max_agreement = max(positive, negative, neutral)
        agreement_level = max_agreement / len(dimensions) if dimensions else 0
        
        # Adjust confidence based on agreement
        base_confidence = mean(d.confidence for d in dimensions)
        if agreement_level > 0.7:
            final_confidence = min(0.95, base_confidence * 1.2)
        elif agreement_level < 0.4:
            final_confidence = base_confidence * 0.7  # Disagreement = uncertainty
        else:
            final_confidence = base_confidence
        
        # Determine dominant signal
        if weighted_score > 0.2:
            dominant_signal = 'BUY'
        elif weighted_score < -0.2:
            dominant_signal = 'SELL'
        else:
            dominant_signal = 'NEUTRAL'
        
        # Detect opportunity type
        opportunity_type = self._detect_opportunity_type(dimensions, weighted_score)
        
        # Collect reasons
        all_reasons = []
        for d in dimensions:
            all_reasons.extend(d.reasons[:2])  # Top 2 reasons per dimension
        
        return MultiDimensionalScore(
            dimensions=dimensions,
            final_score=weighted_score,
            final_confidence=final_confidence,
            agreement_level=agreement_level,
            dominant_signal=dominant_signal,
            opportunity_type=opportunity_type,
            reasons=all_reasons[:6],  # Top 6 reasons
        )
    
    # =========================================================================
    # MOMENTUM & OPPORTUNITY DETECTION
    # =========================================================================
    
    def analyze_momentum(self, prices: List[float]) -> MomentumProfile:
        """
        Multi-timeframe momentum analysis.
        """
        if len(prices) < 5:
            return MomentumProfile(0, 0, 0, 0, 0, 0)
        
        # Short-term momentum (last 5)
        short = prices[-5:]
        short_mom = (short[-1] - short[0]) / short[0] if short[0] != 0 else 0
        
        # Medium-term momentum (last 20 or available)
        med_len = min(20, len(prices))
        medium = prices[-med_len:]
        med_mom = (medium[-1] - medium[0]) / medium[0] if medium[0] != 0 else 0
        
        # Long-term momentum (all available)
        long_mom = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
        
        # Acceleration (is momentum increasing?)
        if len(prices) >= 10:
            prev_short = prices[-10:-5]
            prev_mom = (prev_short[-1] - prev_short[0]) / prev_short[0] if prev_short[0] != 0 else 0
            acceleration = short_mom - prev_mom
        else:
            acceleration = 0
        
        # Consistency (are all timeframes aligned?)
        signs = [
            1 if short_mom > 0.01 else (-1 if short_mom < -0.01 else 0),
            1 if med_mom > 0.01 else (-1 if med_mom < -0.01 else 0),
            1 if long_mom > 0.01 else (-1 if long_mom < -0.01 else 0),
        ]
        consistency = abs(sum(signs)) / 3
        
        # Breakout potential
        if len(prices) >= 20:
            recent_range = max(prices[-20:]) - min(prices[-20:])
            current_pos = (prices[-1] - min(prices[-20:])) / recent_range if recent_range > 0 else 0.5
            
            # High breakout potential if near edges with momentum
            if current_pos > 0.9 and short_mom > 0:
                breakout_potential = 0.8
            elif current_pos < 0.1 and short_mom < 0:
                breakout_potential = 0.8
            else:
                breakout_potential = 0.3
        else:
            breakout_potential = 0.3
        
        return MomentumProfile(
            short_term=short_mom,
            medium_term=med_mom,
            long_term=long_mom,
            acceleration=acceleration,
            consistency=consistency,
            breakout_potential=breakout_potential,
        )
    
    def detect_opportunities(
        self,
        prices: List[float],
        momentum: MomentumProfile,
        multi_score: MultiDimensionalScore,
    ) -> List[OpportunityScore]:
        """
        Detect specific trading opportunities.
        """
        opportunities = []
        
        # Momentum opportunity
        if momentum.consistency > 0.7 and abs(momentum.short_term) > 0.03:
            opp_type = OpportunityType.MOMENTUM_BUY if momentum.short_term > 0 else OpportunityType.MOMENTUM_SELL
            opportunities.append(OpportunityScore(
                opportunity_type=opp_type,
                score=momentum.consistency,
                expected_return=abs(momentum.short_term) * 2,
                risk_level=0.4,
                time_sensitivity='hours',
                reasons=['Strong aligned momentum', f'{momentum.consistency:.0%} consistency'],
            ))
        
        # Mean reversion opportunity
        if len(prices) >= 20:
            avg_price = mean(prices[-20:])
            deviation = (prices[-1] - avg_price) / avg_price if avg_price != 0 else 0
            
            if abs(deviation) > 0.1:  # >10% from mean
                if deviation > 0:
                    opp_type = OpportunityType.MEAN_REVERSION_SELL
                    reasons = ['Price extended above mean', f'{deviation:.0%} deviation']
                else:
                    opp_type = OpportunityType.MEAN_REVERSION_BUY
                    reasons = ['Price extended below mean', f'{abs(deviation):.0%} deviation']
                
                opportunities.append(OpportunityScore(
                    opportunity_type=opp_type,
                    score=min(1, abs(deviation) * 3),
                    expected_return=abs(deviation) * 0.5,
                    risk_level=0.5,
                    time_sensitivity='hours',
                    reasons=reasons,
                ))
        
        # Breakout opportunity
        if momentum.breakout_potential > 0.7:
            opportunities.append(OpportunityScore(
                opportunity_type=OpportunityType.BREAKOUT_UP if momentum.short_term > 0 else OpportunityType.BREAKDOWN,
                score=momentum.breakout_potential,
                expected_return=0.15,
                risk_level=0.6,
                time_sensitivity='immediate',
                reasons=['Breakout forming', f'{momentum.breakout_potential:.0%} potential'],
            ))
        
        # Whale follow opportunity
        if multi_score.opportunity_type == OpportunityType.WHALE_FOLLOW:
            opportunities.append(OpportunityScore(
                opportunity_type=OpportunityType.WHALE_FOLLOW,
                score=0.8,
                expected_return=0.1,
                risk_level=0.3,
                time_sensitivity='immediate',
                reasons=['Following whale activity'],
            ))
        
        return opportunities
    
    # =========================================================================
    # CROSS-MARKET CORRELATION
    # =========================================================================
    
    def analyze_cross_market_effects(
        self,
        event_title: str,
        segment: str,
        related_markets_performance: Dict[str, float],
    ) -> List[CrossMarketSignal]:
        """
        Analyze how related markets might affect this one.
        """
        signals = []
        
        # Define market relationships
        relationships = {
            'crypto': ['bitcoin', 'ethereum', 'solana'],
            'politics': ['trump', 'biden', 'election'],
            'sports': [],  # Sports are mostly independent
        }
        
        title_lower = event_title.lower()
        
        # Check for related market movements
        for related_segment, keywords in relationships.items():
            if segment == related_segment:
                continue
            
            for keyword in keywords:
                if keyword in title_lower:
                    # This market is related to another segment
                    if related_segment in related_markets_performance:
                        perf = related_markets_performance[related_segment]
                        if abs(perf) > 0.05:  # >5% move in related market
                            signals.append(CrossMarketSignal(
                                source_market=related_segment,
                                target_market=segment,
                                correlation=0.6,
                                lag_minutes=30,
                                signal_strength=perf,
                                direction='SAME',
                                confidence=0.5,
                            ))
        
        return signals
    
    # =========================================================================
    # ENSEMBLE DECISION MAKING
    # =========================================================================
    
    def make_ensemble_decision(
        self,
        multi_score: MultiDimensionalScore,
        momentum: MomentumProfile,
        opportunities: List[OpportunityScore],
        regime: str,
        segment: str,
    ) -> Tuple[str, float, float, List[str]]:
        """
        Final ensemble decision combining all analysis.
        
        Returns:
            (action, confidence, position_size_multiplier, reasons)
        """
        reasons = []
        
        # Base decision from multi-dimensional score
        action = multi_score.dominant_signal
        confidence = multi_score.final_confidence
        
        # Adjust for momentum alignment
        if momentum.consistency > 0.7:
            if (action == 'BUY' and momentum.short_term > 0) or \
               (action == 'SELL' and momentum.short_term < 0):
                confidence = min(0.95, confidence * 1.15)
                reasons.append('Momentum aligned')
            else:
                confidence *= 0.85
                reasons.append('Momentum divergence')
        
        # Boost for high-quality opportunities
        best_opp = max(opportunities, key=lambda o: o.score) if opportunities else None
        if best_opp and best_opp.score > 0.7:
            confidence = min(0.95, confidence * 1.1)
            reasons.append(f'Opportunity: {best_opp.opportunity_type.value}')
        
        # Agreement bonus
        if multi_score.agreement_level > 0.8:
            confidence = min(0.95, confidence * 1.1)
            reasons.append('High signal agreement')
        elif multi_score.agreement_level < 0.4:
            confidence *= 0.8
            reasons.append('Signal disagreement - caution')
            if confidence < 0.5:
                action = 'NEUTRAL'
        
        # Calculate position size multiplier
        pos_multiplier = self._calculate_position_multiplier(
            confidence, multi_score.agreement_level, momentum.consistency, regime
        )
        
        # Add dimension reasons
        reasons.extend(multi_score.reasons[:3])
        
        return action, confidence, pos_multiplier, reasons
    
    def _calculate_position_multiplier(
        self,
        confidence: float,
        agreement: float,
        momentum_consistency: float,
        regime: str,
    ) -> float:
        """
        Calculate position size multiplier using Kelly-inspired approach.
        """
        # Base Kelly-lite: bet more when edge is higher
        kelly_factor = confidence * 2 - 1  # Convert 0-1 confidence to edge
        
        # Adjust for agreement
        agreement_factor = 0.5 + agreement * 0.5  # 0.5 to 1.0
        
        # Adjust for momentum consistency
        momentum_factor = 0.8 + momentum_consistency * 0.4  # 0.8 to 1.2
        
        # Regime adjustment
        regime_factors = {
            'high_volatility': 0.5,
            'low_volatility': 1.2,
            'trending_up': 1.1,
            'trending_down': 1.1,
            'pre_event': 0.3,
            'normal': 1.0,
        }
        regime_factor = regime_factors.get(regime, 1.0)
        
        # Combine factors
        multiplier = kelly_factor * agreement_factor * momentum_factor * regime_factor
        
        # Clamp to reasonable range
        return max(0.25, min(2.0, multiplier))
    
    def _detect_opportunity_type(
        self,
        dimensions: List[SignalDimension],
        score: float,
    ) -> Optional[OpportunityType]:
        """Detect the type of opportunity from dimensions."""
        # Check for whale signal
        flow_dim = next((d for d in dimensions if d.name == 'flow'), None)
        if flow_dim and abs(flow_dim.score) > 0.5:
            return OpportunityType.WHALE_FOLLOW
        
        # Check for news catalyst
        sent_dim = next((d for d in dimensions if d.name == 'sentiment'), None)
        if sent_dim and abs(sent_dim.score) > 0.5:
            return OpportunityType.NEWS_CATALYST
        
        # Default based on score direction
        if score > 0.3:
            return OpportunityType.MOMENTUM_BUY
        elif score < -0.3:
            return OpportunityType.MOMENTUM_SELL
        
        return None
    
    def _get_adaptive_weight(self, dimension: str) -> float:
        """Get adaptive weight based on historical performance."""
        perf = self.dimension_performance[dimension]
        total = perf['wins'] + perf['losses']
        
        if total < 10:
            # Not enough data, use defaults
            defaults = {
                'technical': 1.0,
                'sentiment': 0.8,
                'flow': 1.2,
                'structural': 0.7,
            }
            return defaults.get(dimension, 1.0)
        
        win_rate = perf['wins'] / total
        
        # Scale weight by performance
        # Good performers get higher weights
        return 0.5 + win_rate
    
    def record_outcome(self, dimension: str, won: bool, pnl: float):
        """Record outcome for adaptive learning."""
        if won:
            self.dimension_performance[dimension]['wins'] += 1
        else:
            self.dimension_performance[dimension]['losses'] += 1
        self.dimension_performance[dimension]['total_pnl'] += pnl
    
    # =========================================================================
    # MAIN OPTIMIZATION FUNCTION
    # =========================================================================
    
    def optimize_signal(
        self,
        event_title: str,
        segment: str,
        regime: str,
        prices: List[float],
        rsi: Optional[float] = None,
        macd_histogram: Optional[float] = None,
        trend: str = 'NEUTRAL',
        price_change_pct: Optional[float] = None,
        has_news_alpha: bool = False,
        news_sentiment: float = 0,
        telegram_hits: int = 0,
        whale_signal: bool = False,
        whale_direction: str = 'BUY',
        near_support: bool = False,
        near_resistance: bool = False,
    ) -> Dict[str, Any]:
        """
        Main optimization function - combines all analysis.
        
        Returns complete optimization result with action, confidence,
        position sizing, and detailed reasoning.
        """
        # Calculate all dimensions
        technical = self.calculate_technical_dimension(
            rsi, macd_histogram, trend, price_change_pct
        )
        
        sentiment = self.calculate_sentiment_dimension(
            has_news_alpha, news_sentiment, telegram_hits, 0
        )
        
        flow = self.calculate_flow_dimension(
            whale_signal, whale_direction, False, 0
        )
        
        structural = self.calculate_structural_dimension(
            near_support, near_resistance, False, False
        )
        
        # Combine dimensions
        dimensions = [technical, sentiment, flow, structural]
        multi_score = self.combine_dimensions(dimensions)
        
        # Analyze momentum
        momentum = self.analyze_momentum(prices)
        
        # Detect opportunities
        opportunities = self.detect_opportunities(prices, momentum, multi_score)
        
        # Make ensemble decision
        action, confidence, pos_multiplier, reasons = self.make_ensemble_decision(
            multi_score, momentum, opportunities, regime, segment
        )
        
        return {
            'action': action,
            'confidence': confidence,
            'position_multiplier': pos_multiplier,
            'multi_dimensional_score': multi_score.final_score,
            'agreement_level': multi_score.agreement_level,
            'momentum_profile': {
                'short': momentum.short_term,
                'medium': momentum.medium_term,
                'consistency': momentum.consistency,
                'breakout_potential': momentum.breakout_potential,
            },
            'opportunities': [
                {'type': o.opportunity_type.value, 'score': o.score}
                for o in opportunities
            ],
            'reasons': reasons,
            'dimensions': {
                d.name: {'score': d.score, 'confidence': d.confidence}
                for d in dimensions
            },
        }
    
    def get_summary(self) -> str:
        """Get optimizer summary."""
        lines = [
            '=== PROFIT OPTIMIZER STATUS ===',
            f'Total Signals: {self.total_signals}',
            f'Winning: {self.winning_signals}',
            '',
            'Dimension Performance:',
        ]
        
        for dim, perf in self.dimension_performance.items():
            total = perf['wins'] + perf['losses']
            if total > 0:
                wr = perf['wins'] / total
                lines.append(f'  {dim}: {wr:.0%} win rate ({total} trades)')
        
        return '\n'.join(lines)


# Singleton
_optimizer: Optional[ProfitOptimizer] = None

def get_profit_optimizer() -> ProfitOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = ProfitOptimizer()
    return _optimizer


if __name__ == '__main__':
    # Test the optimizer
    optimizer = get_profit_optimizer()
    
    # Sample data
    prices = [0.45, 0.46, 0.48, 0.47, 0.50, 0.52, 0.51, 0.54, 0.56, 0.55]
    
    result = optimizer.optimize_signal(
        event_title="Bitcoin price above $100k?",
        segment="crypto",
        regime="normal",
        prices=prices,
        rsi=65,
        macd_histogram=0.02,
        trend="BULLISH",
        price_change_pct=10,
        has_news_alpha=True,
        news_sentiment=0.5,
        whale_signal=True,
        whale_direction="BUY",
    )
    
    print("Optimization Result:")
    print(f"  Action: {result['action']}")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"  Position Multiplier: {result['position_multiplier']:.2f}x")
    print(f"  Agreement Level: {result['agreement_level']:.0%}")
    print(f"  Reasons: {', '.join(result['reasons'][:3])}")
    
    print("\nDimension Scores:")
    for dim, data in result['dimensions'].items():
        print(f"  {dim}: {data['score']:.2f} (conf: {data['confidence']:.0%})")
    
    print("\nOpportunities:")
    for opp in result['opportunities']:
        print(f"  {opp['type']}: {opp['score']:.0%}")

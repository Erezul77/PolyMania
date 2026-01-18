"""
Risk Manager - Dynamic Risk & Portfolio Optimization

Manages all aspects of risk:

1. POSITION LIMITS
   - Max position per event
   - Max exposure per segment
   - Correlation-adjusted limits

2. DRAWDOWN PROTECTION
   - Daily loss limits
   - Equity curve protection
   - Automatic risk reduction

3. PORTFOLIO OPTIMIZATION
   - Diversification scoring
   - Correlation monitoring
   - Rebalancing triggers

4. DYNAMIC PARAMETERS
   - Adaptive stop-losses
   - Dynamic take-profits
   - Regime-based adjustments

5. RISK METRICS
   - Sharpe ratio tracking
   - Max drawdown monitoring
   - Value at Risk (VaR)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
from statistics import mean, stdev
import math
import json
import os

logger = logging.getLogger('polymania.risk')


class RiskLevel(Enum):
    AGGRESSIVE = 'aggressive'
    NORMAL = 'normal'
    CONSERVATIVE = 'conservative'
    DEFENSIVE = 'defensive'
    EMERGENCY = 'emergency'


@dataclass
class Position:
    """Active position"""
    event_id: str
    segment: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    current_price: float
    size: float
    entry_time: datetime
    unrealized_pnl: float = 0


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    total_exposure: float
    segment_exposures: Dict[str, float]
    position_count: int
    largest_position: float
    correlation_risk: float
    portfolio_heat: float  # 0 to 1
    var_95: float  # 95% VaR
    current_drawdown: float
    sharpe_estimate: float


@dataclass
class RiskAdjustment:
    """Recommended risk adjustments"""
    position_size_multiplier: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    should_trade: bool
    reason: str


class RiskManager:
    """
    Comprehensive risk management system.
    """
    
    # Risk level configurations
    RISK_CONFIGS = {
        RiskLevel.AGGRESSIVE: {
            'max_position': 0.20,
            'max_segment_exposure': 0.40,
            'max_total_exposure': 0.80,
            'max_daily_loss': 0.15,
            'stop_loss_mult': 0.8,
            'take_profit_mult': 1.3,
        },
        RiskLevel.NORMAL: {
            'max_position': 0.12,
            'max_segment_exposure': 0.30,
            'max_total_exposure': 0.60,
            'max_daily_loss': 0.10,
            'stop_loss_mult': 1.0,
            'take_profit_mult': 1.0,
        },
        RiskLevel.CONSERVATIVE: {
            'max_position': 0.08,
            'max_segment_exposure': 0.20,
            'max_total_exposure': 0.40,
            'max_daily_loss': 0.07,
            'stop_loss_mult': 1.2,
            'take_profit_mult': 0.8,
        },
        RiskLevel.DEFENSIVE: {
            'max_position': 0.05,
            'max_segment_exposure': 0.15,
            'max_total_exposure': 0.25,
            'max_daily_loss': 0.05,
            'stop_loss_mult': 1.5,
            'take_profit_mult': 0.6,
        },
        RiskLevel.EMERGENCY: {
            'max_position': 0.02,
            'max_segment_exposure': 0.05,
            'max_total_exposure': 0.10,
            'max_daily_loss': 0.02,
            'stop_loss_mult': 2.0,
            'take_profit_mult': 0.4,
        },
    }
    
    def __init__(self):
        # Current positions
        self.positions: Dict[str, Position] = {}
        
        # Performance history
        self.daily_pnl: Dict[str, float] = {}  # date -> pnl
        self.peak_equity = 1.0
        self.current_equity = 1.0
        
        # Risk state
        self.current_risk_level = RiskLevel.NORMAL
        self.risk_level_changes: List[Tuple[datetime, RiskLevel, str]] = []
        
        # Metrics history
        self.metrics_history: List[Tuple[datetime, RiskMetrics]] = []
        
        # Correlation tracking
        self.segment_correlations: Dict[Tuple[str, str], float] = {}
        
        # Data file
        self.data_file = 'data/risk_manager.json'
        self._load_state()
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def add_position(
        self,
        event_id: str,
        segment: str,
        direction: str,
        entry_price: float,
        size: float,
    ):
        """Add a new position."""
        self.positions[event_id] = Position(
            event_id=event_id,
            segment=segment,
            direction=direction,
            entry_price=entry_price,
            current_price=entry_price,
            size=size,
            entry_time=datetime.now(timezone.utc),
        )
        self._save_state()
    
    def update_position(self, event_id: str, current_price: float):
        """Update position with current price."""
        if event_id in self.positions:
            pos = self.positions[event_id]
            pos.current_price = current_price
            
            # Calculate unrealized PnL
            if pos.direction == 'LONG':
                pos.unrealized_pnl = (current_price - pos.entry_price) / pos.entry_price * pos.size
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) / pos.entry_price * pos.size
    
    def close_position(self, event_id: str, exit_price: float) -> Optional[float]:
        """Close a position and return realized PnL."""
        if event_id not in self.positions:
            return None
        
        pos = self.positions[event_id]
        
        # Calculate realized PnL
        if pos.direction == 'LONG':
            pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.size
        else:
            pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.size
        
        # Update equity
        self.current_equity += pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        # Update daily PnL
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self.daily_pnl[today] = self.daily_pnl.get(today, 0) + pnl
        
        # Remove position
        del self.positions[event_id]
        
        # Check if risk level needs adjustment
        self._update_risk_level()
        
        self._save_state()
        return pnl
    
    # =========================================================================
    # RISK METRICS
    # =========================================================================
    
    def calculate_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics."""
        # Total exposure
        total_exposure = sum(p.size for p in self.positions.values())
        
        # Segment exposures
        segment_exposures = defaultdict(float)
        for pos in self.positions.values():
            segment_exposures[pos.segment] += pos.size
        
        # Largest position
        largest = max((p.size for p in self.positions.values()), default=0)
        
        # Correlation risk
        correlation_risk = self._calculate_correlation_risk()
        
        # Portfolio heat (how much risk we're taking)
        config = self.RISK_CONFIGS[self.current_risk_level]
        portfolio_heat = total_exposure / config['max_total_exposure'] if config['max_total_exposure'] > 0 else 0
        
        # VaR estimate (simplified)
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        var_95 = self._estimate_var()
        
        # Current drawdown
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # Sharpe estimate
        sharpe = self._estimate_sharpe()
        
        metrics = RiskMetrics(
            total_exposure=total_exposure,
            segment_exposures=dict(segment_exposures),
            position_count=len(self.positions),
            largest_position=largest,
            correlation_risk=correlation_risk,
            portfolio_heat=portfolio_heat,
            var_95=var_95,
            current_drawdown=drawdown,
            sharpe_estimate=sharpe,
        )
        
        # Store metrics
        self.metrics_history.append((datetime.now(timezone.utc), metrics))
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate risk from correlated positions."""
        if len(self.positions) < 2:
            return 0
        
        segments = list(set(p.segment for p in self.positions.values()))
        if len(segments) < 2:
            return 0.5  # All in one segment = some risk
        
        # Simple correlation estimation based on segments
        # In reality, would need historical price correlation
        high_corr_pairs = [
            ('crypto', 'crypto'),
            ('politics', 'politics'),
        ]
        
        corr_score = 0
        for seg1 in segments:
            for seg2 in segments:
                if seg1 != seg2:
                    if (seg1, seg2) in high_corr_pairs or (seg2, seg1) in high_corr_pairs:
                        corr_score += 0.2
        
        return min(1, corr_score)
    
    def _estimate_var(self) -> float:
        """Estimate 95% Value at Risk."""
        if len(self.daily_pnl) < 10:
            return 0.05  # Default 5%
        
        daily_returns = list(self.daily_pnl.values())[-30:]  # Last 30 days
        if not daily_returns:
            return 0.05
        
        try:
            avg = mean(daily_returns)
            std = stdev(daily_returns) if len(daily_returns) > 1 else 0.02
            var_95 = avg - 1.645 * std  # 95% confidence
            return abs(var_95)
        except Exception:
            return 0.05
    
    def _estimate_sharpe(self) -> float:
        """Estimate Sharpe ratio."""
        if len(self.daily_pnl) < 20:
            return 0
        
        daily_returns = list(self.daily_pnl.values())[-60:]  # Last 60 days
        if not daily_returns:
            return 0
        
        try:
            avg = mean(daily_returns)
            std = stdev(daily_returns) if len(daily_returns) > 1 else 0.01
            if std == 0:
                return 0
            
            # Annualized Sharpe (assuming 365 trading days)
            sharpe = (avg * 365) / (std * math.sqrt(365))
            return sharpe
        except Exception:
            return 0
    
    # =========================================================================
    # RISK LEVEL MANAGEMENT
    # =========================================================================
    
    def _update_risk_level(self):
        """Automatically adjust risk level based on performance."""
        metrics = self.calculate_metrics()
        old_level = self.current_risk_level
        reason = ""
        
        # Check drawdown
        if metrics.current_drawdown > 0.20:
            self.current_risk_level = RiskLevel.EMERGENCY
            reason = f"Large drawdown: {metrics.current_drawdown:.0%}"
        elif metrics.current_drawdown > 0.15:
            self.current_risk_level = RiskLevel.DEFENSIVE
            reason = f"Significant drawdown: {metrics.current_drawdown:.0%}"
        elif metrics.current_drawdown > 0.10:
            self.current_risk_level = RiskLevel.CONSERVATIVE
            reason = f"Moderate drawdown: {metrics.current_drawdown:.0%}"
        
        # Check daily loss
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        daily_loss = self.daily_pnl.get(today, 0)
        config = self.RISK_CONFIGS[self.current_risk_level]
        
        if daily_loss < -config['max_daily_loss']:
            if self.current_risk_level.value not in ['defensive', 'emergency']:
                self.current_risk_level = RiskLevel.DEFENSIVE
                reason = f"Daily loss limit hit: {daily_loss:.0%}"
        
        # Check Sharpe and performance for potential upgrades
        if metrics.sharpe_estimate > 2 and metrics.current_drawdown < 0.05:
            if self.current_risk_level in [RiskLevel.CONSERVATIVE, RiskLevel.NORMAL]:
                self.current_risk_level = RiskLevel.AGGRESSIVE
                reason = f"Strong performance: Sharpe {metrics.sharpe_estimate:.1f}"
        elif metrics.sharpe_estimate > 1 and metrics.current_drawdown < 0.08:
            if self.current_risk_level == RiskLevel.CONSERVATIVE:
                self.current_risk_level = RiskLevel.NORMAL
                reason = f"Good performance: Sharpe {metrics.sharpe_estimate:.1f}"
        
        # Log changes
        if old_level != self.current_risk_level:
            self.risk_level_changes.append(
                (datetime.now(timezone.utc), self.current_risk_level, reason)
            )
            logger.info(f"Risk level changed: {old_level.value} -> {self.current_risk_level.value} ({reason})")
    
    def set_risk_level(self, level: RiskLevel, reason: str = "Manual override"):
        """Manually set risk level."""
        old_level = self.current_risk_level
        self.current_risk_level = level
        self.risk_level_changes.append((datetime.now(timezone.utc), level, reason))
        logger.info(f"Risk level manually set: {old_level.value} -> {level.value}")
        self._save_state()
    
    # =========================================================================
    # RISK ADJUSTMENTS
    # =========================================================================
    
    def get_risk_adjustment(
        self,
        event_id: str,
        segment: str,
        proposed_size: float,
    ) -> RiskAdjustment:
        """
        Get risk adjustments for a proposed trade.
        """
        metrics = self.calculate_metrics()
        config = self.RISK_CONFIGS[self.current_risk_level]
        
        reasons = []
        should_trade = True
        size_mult = 1.0
        
        # Check position limit
        if proposed_size > config['max_position']:
            size_mult = config['max_position'] / proposed_size
            reasons.append(f"Position capped at {config['max_position']:.0%}")
        
        # Check segment exposure
        current_segment_exposure = metrics.segment_exposures.get(segment, 0)
        if current_segment_exposure + proposed_size > config['max_segment_exposure']:
            remaining = config['max_segment_exposure'] - current_segment_exposure
            if remaining <= 0:
                should_trade = False
                reasons.append(f"Segment {segment} at limit")
            else:
                size_mult = min(size_mult, remaining / proposed_size)
                reasons.append(f"Segment exposure limited")
        
        # Check total exposure
        if metrics.total_exposure + proposed_size > config['max_total_exposure']:
            remaining = config['max_total_exposure'] - metrics.total_exposure
            if remaining <= 0:
                should_trade = False
                reasons.append("Total exposure at limit")
            else:
                size_mult = min(size_mult, remaining / proposed_size)
                reasons.append("Total exposure limited")
        
        # Check daily loss limit
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        daily_loss = self.daily_pnl.get(today, 0)
        if daily_loss < -config['max_daily_loss'] * 0.8:  # 80% of daily limit
            size_mult *= 0.5
            reasons.append("Approaching daily loss limit")
        
        # Portfolio heat adjustment
        if metrics.portfolio_heat > 0.8:
            size_mult *= 0.5
            reasons.append("High portfolio heat")
        
        # Correlation adjustment
        if metrics.correlation_risk > 0.6:
            size_mult *= 0.8
            reasons.append("High correlation risk")
        
        reason = "; ".join(reasons) if reasons else "Normal"
        
        return RiskAdjustment(
            position_size_multiplier=size_mult,
            stop_loss_multiplier=config['stop_loss_mult'],
            take_profit_multiplier=config['take_profit_mult'],
            should_trade=should_trade,
            reason=reason,
        )
    
    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================
    
    def _save_state(self):
        """Save state to disk."""
        try:
            os.makedirs('data', exist_ok=True)
            state = {
                'daily_pnl': self.daily_pnl,
                'peak_equity': self.peak_equity,
                'current_equity': self.current_equity,
                'risk_level': self.current_risk_level.value,
                'positions': {
                    k: {
                        'event_id': v.event_id,
                        'segment': v.segment,
                        'direction': v.direction,
                        'entry_price': v.entry_price,
                        'current_price': v.current_price,
                        'size': v.size,
                        'entry_time': v.entry_time.isoformat(),
                    }
                    for k, v in self.positions.items()
                },
            }
            with open(self.data_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f'Failed to save risk state: {e}')
    
    def _load_state(self):
        """Load state from disk."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    state = json.load(f)
                
                self.daily_pnl = state.get('daily_pnl', {})
                self.peak_equity = state.get('peak_equity', 1.0)
                self.current_equity = state.get('current_equity', 1.0)
                
                try:
                    self.current_risk_level = RiskLevel(state.get('risk_level', 'normal'))
                except ValueError:
                    self.current_risk_level = RiskLevel.NORMAL
                
                # Load positions
                for k, v in state.get('positions', {}).items():
                    self.positions[k] = Position(
                        event_id=v['event_id'],
                        segment=v['segment'],
                        direction=v['direction'],
                        entry_price=v['entry_price'],
                        current_price=v['current_price'],
                        size=v['size'],
                        entry_time=datetime.fromisoformat(v['entry_time']),
                    )
        except Exception as e:
            logger.warning(f'Failed to load risk state: {e}')
    
    def get_summary(self) -> str:
        """Get risk manager summary."""
        metrics = self.calculate_metrics()
        config = self.RISK_CONFIGS[self.current_risk_level]
        
        lines = [
            '=== RISK MANAGER STATUS ===',
            '',
            f'Risk Level: {self.current_risk_level.value.upper()}',
            f'Portfolio Heat: {metrics.portfolio_heat:.0%}',
            '',
            'Current Exposure:',
            f'  Total: {metrics.total_exposure:.1%} / {config["max_total_exposure"]:.0%}',
            f'  Positions: {metrics.position_count}',
            f'  Largest: {metrics.largest_position:.1%}',
            '',
            'By Segment:',
        ]
        
        for seg, exp in metrics.segment_exposures.items():
            lines.append(f'  {seg}: {exp:.1%}')
        
        lines.extend([
            '',
            'Risk Metrics:',
            f'  Current Drawdown: {metrics.current_drawdown:.1%}',
            f'  VaR (95%): {metrics.var_95:.1%}',
            f'  Correlation Risk: {metrics.correlation_risk:.0%}',
            f'  Sharpe Estimate: {metrics.sharpe_estimate:.2f}',
            '',
            f'Equity: {self.current_equity:.2f} (Peak: {self.peak_equity:.2f})',
        ])
        
        # Recent risk level changes
        if self.risk_level_changes:
            lines.append('\nRecent Risk Changes:')
            for dt, level, reason in self.risk_level_changes[-3:]:
                lines.append(f'  {dt.strftime("%Y-%m-%d %H:%M")}: {level.value} - {reason}')
        
        return '\n'.join(lines)


# Singleton
_manager: Optional[RiskManager] = None

def get_risk_manager() -> RiskManager:
    global _manager
    if _manager is None:
        _manager = RiskManager()
    return _manager

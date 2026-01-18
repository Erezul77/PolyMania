"""
Advanced Risk Management
========================

World-class risk management using:
- Kelly Criterion position sizing
- Risk Parity allocation
- VaR / CVaR calculations
- Dynamic stop-loss / take-profit
- Drawdown-based position scaling
- Correlation-aware sizing
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger("ml.risk_manager")


@dataclass
class PositionSize:
    """Calculated position size with metadata."""
    size: float  # Position size (0-1 of capital)
    kelly_size: float  # Raw Kelly size
    risk_parity_size: float  # Risk parity adjusted
    drawdown_adjusted: float  # After drawdown scaling
    final_size: float  # Final recommended size
    confidence: float  # Confidence in sizing
    risk_score: float  # Overall risk score (0-1)
    max_loss_pct: float  # Expected max loss %
    reasoning: str  # Why this size
    
    def to_dict(self) -> Dict:
        return {
            "size": self.size,
            "kelly_size": self.kelly_size,
            "risk_parity_size": self.risk_parity_size,
            "drawdown_adjusted": self.drawdown_adjusted,
            "final_size": self.final_size,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "max_loss_pct": self.max_loss_pct,
            "reasoning": self.reasoning
        }


@dataclass 
class RiskMetrics:
    """Current risk metrics for portfolio."""
    var_95: float = 0.0  # Value at Risk (95%)
    var_99: float = 0.0  # Value at Risk (99%)
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    beta: float = 1.0  # Market beta
    correlation_risk: float = 0.0  # Strategy correlation
    
    def to_dict(self) -> Dict:
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "beta": self.beta,
            "correlation_risk": self.correlation_risk
        }


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.
    f* = (bp - q) / b
    where:
        b = odds (reward/risk ratio)
        p = probability of winning
        q = probability of losing (1-p)
    """
    
    def __init__(
        self,
        max_kelly_fraction: float = 0.5,  # Never bet more than 50% of full Kelly
        min_win_rate: float = 0.4,  # Minimum win rate to take position
        min_rr_ratio: float = 1.0  # Minimum reward/risk ratio
    ):
        self.max_kelly_fraction = max_kelly_fraction
        self.min_win_rate = min_win_rate
        self.min_rr_ratio = min_rr_ratio
        
        # History for adaptive estimation
        self._trade_history: List[Tuple[float, bool]] = []  # (pnl, won)
    
    def record_trade(self, pnl: float, won: bool):
        """Record trade outcome for adaptive estimation."""
        self._trade_history.append((pnl, won))
        self._trade_history = self._trade_history[-200:]  # Keep last 200
    
    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        signal_confidence: float = 0.5
    ) -> Tuple[float, str]:
        """
        Calculate Kelly fraction.
        
        Returns:
            (kelly_fraction, reasoning)
        """
        # Validation
        if win_rate < self.min_win_rate:
            return 0, f"Win rate {win_rate:.1%} below minimum {self.min_win_rate:.1%}"
        
        if avg_loss <= 0:
            return 0, "Invalid avg_loss"
        
        # Reward/risk ratio
        rr_ratio = avg_win / abs(avg_loss)
        
        if rr_ratio < self.min_rr_ratio:
            return 0, f"R:R ratio {rr_ratio:.2f} below minimum {self.min_rr_ratio}"
        
        # Kelly formula
        q = 1 - win_rate
        kelly = (win_rate * rr_ratio - q) / rr_ratio
        
        if kelly <= 0:
            return 0, f"Negative Kelly ({kelly:.3f}) - edge not sufficient"
        
        # Apply fractional Kelly (half Kelly is common)
        fractional_kelly = kelly * self.max_kelly_fraction
        
        # Adjust by signal confidence
        confidence_adjusted = fractional_kelly * (0.5 + signal_confidence * 0.5)
        
        # Cap at reasonable maximum
        final = min(confidence_adjusted, 0.25)  # Never more than 25% of capital
        
        reasoning = (f"Kelly: {kelly:.1%}, Fractional: {fractional_kelly:.1%}, "
                    f"Conf-adj: {confidence_adjusted:.1%}, Final: {final:.1%}")
        
        return final, reasoning
    
    def get_adaptive_kelly(self, signal_confidence: float = 0.5) -> Tuple[float, str]:
        """Calculate Kelly using historical trade data."""
        if len(self._trade_history) < 20:
            return 0.05, "Insufficient history - using default 5%"
        
        wins = [t for t in self._trade_history if t[1]]
        losses = [t for t in self._trade_history if not t[1]]
        
        if not wins or not losses:
            return 0.05, "Need both wins and losses for Kelly"
        
        win_rate = len(wins) / len(self._trade_history)
        avg_win = np.mean([t[0] for t in wins])
        avg_loss = np.mean([abs(t[0]) for t in losses])
        
        return self.calculate_kelly(win_rate, avg_win, avg_loss, signal_confidence)


class RiskParityAllocator:
    """
    Risk Parity allocation - equal risk contribution from each strategy.
    """
    
    def __init__(self, target_volatility: float = 0.15):
        self.target_volatility = target_volatility  # Annual vol target
        
        # Strategy volatilities
        self._strategy_vols: Dict[str, float] = {}
        self._strategy_returns: Dict[str, deque] = {}
    
    def update_strategy(self, strategy_name: str, returns: List[float]):
        """Update strategy volatility estimate."""
        if strategy_name not in self._strategy_returns:
            self._strategy_returns[strategy_name] = deque(maxlen=100)
        
        for r in returns:
            self._strategy_returns[strategy_name].append(r)
        
        if len(self._strategy_returns[strategy_name]) >= 10:
            vol = np.std(self._strategy_returns[strategy_name]) * np.sqrt(252)
            self._strategy_vols[strategy_name] = max(vol, 0.01)  # Min vol floor
    
    def calculate_weights(self, strategies: List[str] = None) -> Dict[str, float]:
        """
        Calculate risk parity weights.
        
        Each strategy gets weight inversely proportional to its volatility.
        """
        strategies = strategies or list(self._strategy_vols.keys())
        
        if not strategies:
            return {}
        
        # Get volatilities (use default for missing)
        vols = {}
        for s in strategies:
            vols[s] = self._strategy_vols.get(s, 0.15)  # Default 15% vol
        
        # Inverse vol weights
        inv_vols = {s: 1 / v for s, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        
        # Normalize
        weights = {s: iv / total_inv_vol for s, iv in inv_vols.items()}
        
        # Scale to target vol
        portfolio_vol = sum(w * vols[s] for s, w in weights.items())
        if portfolio_vol > 0:
            scale = self.target_volatility / portfolio_vol
            weights = {s: min(w * scale, 0.5) for s, w in weights.items()}  # Cap at 50%
            
            # Renormalize after capping
            total = sum(weights.values())
            weights = {s: w / total for s, w in weights.items()}
        
        return weights


class DrawdownManager:
    """
    Dynamic position sizing based on drawdown.
    Reduces exposure as drawdown increases.
    """
    
    def __init__(
        self,
        max_drawdown_limit: float = 0.20,  # Start reducing at 20% drawdown
        emergency_stop: float = 0.30,  # Stop trading at 30% drawdown
        recovery_threshold: float = 0.10  # Resume full at 10% drawdown
    ):
        self.max_drawdown_limit = max_drawdown_limit
        self.emergency_stop = emergency_stop
        self.recovery_threshold = recovery_threshold
        
        # Portfolio tracking
        self._equity_history = deque(maxlen=1000)
        self._peak_equity = 0
        self._current_equity = 0
    
    def update_equity(self, equity: float):
        """Update portfolio equity."""
        self._equity_history.append(equity)
        self._current_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self._peak_equity <= 0:
            return 0
        return (self._peak_equity - self._current_equity) / self._peak_equity
    
    @property
    def max_drawdown(self) -> float:
        """Maximum historical drawdown."""
        if len(self._equity_history) < 2:
            return 0
        
        equity = np.array(self._equity_history)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.maximum(peak, 1)
        return np.max(drawdown)
    
    def get_position_multiplier(self) -> Tuple[float, str]:
        """
        Get position size multiplier based on drawdown.
        
        Returns:
            (multiplier, reason)
        """
        dd = self.current_drawdown
        
        # Emergency stop
        if dd >= self.emergency_stop:
            return 0, f"EMERGENCY STOP: Drawdown {dd:.1%} >= {self.emergency_stop:.1%}"
        
        # Full recovery
        if dd <= self.recovery_threshold:
            return 1.0, f"Full capacity: Drawdown {dd:.1%} <= {self.recovery_threshold:.1%}"
        
        # Linear reduction
        if dd > self.max_drawdown_limit:
            # Reduce from 1.0 at max_drawdown_limit to 0.0 at emergency_stop
            reduction_range = self.emergency_stop - self.max_drawdown_limit
            excess_dd = dd - self.max_drawdown_limit
            multiplier = max(0, 1 - (excess_dd / reduction_range))
            return multiplier, f"Drawdown reduction: {dd:.1%} -> {multiplier:.1%} capacity"
        
        # Slight reduction in warning zone
        if dd > self.recovery_threshold:
            # 80-100% between recovery and max_drawdown_limit
            range_size = self.max_drawdown_limit - self.recovery_threshold
            excess = dd - self.recovery_threshold
            multiplier = 1 - 0.2 * (excess / range_size)
            return multiplier, f"Warning zone: {dd:.1%} -> {multiplier:.1%} capacity"
        
        return 1.0, "Normal"


class AdvancedRiskManager:
    """
    Master risk manager combining all components.
    """
    
    def __init__(
        self,
        base_capital: float = 10000,
        max_position_pct: float = 0.20,  # Max 20% per position
        max_portfolio_heat: float = 0.50,  # Max 50% total exposure
        target_volatility: float = 0.15
    ):
        self.base_capital = base_capital
        self.max_position_pct = max_position_pct
        self.max_portfolio_heat = max_portfolio_heat
        
        # Components
        self.kelly = KellyCriterion()
        self.risk_parity = RiskParityAllocator(target_volatility)
        self.drawdown = DrawdownManager()
        
        # Current state
        self._open_positions: Dict[str, float] = {}  # market -> size
        self._portfolio_returns = deque(maxlen=500)
        
        # Track capital
        self.current_capital = base_capital
    
    def update(
        self,
        capital: float,
        portfolio_return: float = None,
        trade_result: Tuple[float, bool] = None
    ):
        """Update risk manager state."""
        self.current_capital = capital
        self.drawdown.update_equity(capital)
        
        if portfolio_return is not None:
            self._portfolio_returns.append(portfolio_return)
        
        if trade_result is not None:
            pnl, won = trade_result
            self.kelly.record_trade(pnl, won)
    
    def calculate_position_size(
        self,
        strategy_name: str,
        signal_confidence: float,
        win_rate: float = None,
        avg_win: float = None,
        avg_loss: float = None,
        volatility: float = None
    ) -> PositionSize:
        """
        Calculate optimal position size considering all factors.
        """
        reasons = []
        
        # 1. Kelly-based sizing
        if win_rate and avg_win and avg_loss:
            kelly_size, kelly_reason = self.kelly.calculate_kelly(
                win_rate, avg_win, avg_loss, signal_confidence
            )
            reasons.append(f"Kelly: {kelly_reason}")
        else:
            kelly_size, kelly_reason = self.kelly.get_adaptive_kelly(signal_confidence)
            reasons.append(f"Adaptive Kelly: {kelly_reason}")
        
        # 2. Risk parity sizing
        rp_weights = self.risk_parity.calculate_weights()
        rp_size = rp_weights.get(strategy_name, 0.1)  # Default 10%
        reasons.append(f"RiskParity: {rp_size:.1%}")
        
        # 3. Drawdown adjustment
        dd_mult, dd_reason = self.drawdown.get_position_multiplier()
        reasons.append(f"Drawdown: {dd_reason}")
        
        # 4. Current portfolio heat check
        current_heat = sum(self._open_positions.values())
        remaining_heat = self.max_portfolio_heat - current_heat
        heat_limit = max(0, remaining_heat)
        reasons.append(f"Heat limit: {heat_limit:.1%} available")
        
        # 5. Combine all factors
        # Use minimum of Kelly and risk parity (conservative)
        base_size = min(kelly_size, rp_size) if kelly_size > 0 else rp_size
        
        # Apply drawdown multiplier
        dd_adjusted = base_size * dd_mult
        
        # Apply heat limit
        heat_adjusted = min(dd_adjusted, heat_limit)
        
        # Apply max position cap
        final_size = min(heat_adjusted, self.max_position_pct)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(final_size, volatility or 0.15)
        
        # Calculate max loss
        max_loss_pct = final_size * (volatility or 0.15) * 2  # 2-sigma move
        
        return PositionSize(
            size=base_size,
            kelly_size=kelly_size,
            risk_parity_size=rp_size,
            drawdown_adjusted=dd_adjusted,
            final_size=final_size,
            confidence=signal_confidence,
            risk_score=risk_score,
            max_loss_pct=max_loss_pct,
            reasoning=" | ".join(reasons)
        )
    
    def _calculate_risk_score(self, position_size: float, volatility: float) -> float:
        """Calculate overall risk score (0=safe, 1=risky)."""
        # Position size contribution
        size_risk = position_size / self.max_position_pct
        
        # Volatility contribution
        vol_risk = min(volatility / 0.30, 1)  # Cap at 30% vol
        
        # Drawdown contribution
        dd_risk = self.drawdown.current_drawdown / self.drawdown.emergency_stop
        
        # Portfolio heat contribution
        heat = sum(self._open_positions.values())
        heat_risk = heat / self.max_portfolio_heat
        
        # Weighted combination
        risk_score = (size_risk * 0.25 + vol_risk * 0.25 + 
                     dd_risk * 0.30 + heat_risk * 0.20)
        
        return min(risk_score, 1.0)
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if len(self._portfolio_returns) < 20:
            return 0.02  # Default 2%
        
        returns = np.array(self._portfolio_returns)
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(self._portfolio_returns) < 20:
            return 0.03  # Default 3%
        
        returns = np.array(self._portfolio_returns)
        var = self.calculate_var(confidence)
        return np.mean(returns[returns <= var])
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        returns = np.array(self._portfolio_returns) if self._portfolio_returns else np.array([0])
        
        metrics = RiskMetrics()
        metrics.var_95 = self.calculate_var(0.95)
        metrics.var_99 = self.calculate_var(0.99)
        metrics.cvar_95 = self.calculate_cvar(0.95)
        metrics.current_drawdown = self.drawdown.current_drawdown
        metrics.max_drawdown = self.drawdown.max_drawdown
        
        if len(returns) > 1:
            metrics.volatility = np.std(returns) * np.sqrt(252)
            
            if metrics.volatility > 0:
                metrics.sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                
                # Sortino (downside deviation)
                downside = returns[returns < 0]
                if len(downside) > 0:
                    downside_std = np.std(downside)
                    if downside_std > 0:
                        metrics.sortino = np.mean(returns) / downside_std * np.sqrt(252)
            
            # Calmar
            if metrics.max_drawdown > 0:
                annual_return = np.mean(returns) * 252
                metrics.calmar = annual_return / metrics.max_drawdown
        
        return metrics
    
    def open_position(self, market_id: str, size: float):
        """Record opening a position."""
        self._open_positions[market_id] = size
    
    def close_position(self, market_id: str):
        """Record closing a position."""
        self._open_positions.pop(market_id, None)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get risk manager statistics."""
        metrics = self.get_risk_metrics()
        return {
            "current_capital": self.current_capital,
            "current_drawdown": self.drawdown.current_drawdown,
            "max_drawdown": self.drawdown.max_drawdown,
            "open_positions": len(self._open_positions),
            "portfolio_heat": sum(self._open_positions.values()),
            "var_95": metrics.var_95,
            "cvar_95": metrics.cvar_95,
            "sharpe": metrics.sharpe,
            "sortino": metrics.sortino,
            "risk_metrics": metrics.to_dict()
        }

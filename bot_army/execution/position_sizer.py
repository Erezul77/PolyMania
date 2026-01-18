"""Position sizing algorithms."""

import numpy as np
import logging
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

from ..core.events import Signal

logger = logging.getLogger("execution.sizing")


@dataclass
class SizingResult:
    """Position sizing result."""
    quantity: float
    value: float
    risk_amount: float
    method: str
    confidence: float
    metadata: Dict[str, Any] = None


class PositionSizer:
    """
    Advanced position sizing algorithms.
    
    Methods:
    - Fixed fractional
    - Kelly criterion
    - Volatility-based
    - Risk parity
    - Optimal f
    """
    
    def __init__(
        self,
        portfolio_value: float = 10000,
        max_position_pct: float = 0.15,
        max_risk_pct: float = 0.02,
        default_method: str = "kelly"
    ):
        self.portfolio_value = portfolio_value
        self.max_position_pct = max_position_pct
        self.max_risk_pct = max_risk_pct
        self.default_method = default_method
        
        # Historical data for adaptive sizing
        self._win_history: list = []
        self._loss_history: list = []
    
    def calculate(
        self,
        signal: Signal,
        method: str = None,
        volatility: float = None,
        **kwargs
    ) -> SizingResult:
        """Calculate position size."""
        method = method or self.default_method
        
        if method == "fixed":
            return self._fixed_fractional(signal, kwargs.get("fraction", 0.1))
        elif method == "kelly":
            return self._kelly_criterion(signal)
        elif method == "volatility":
            return self._volatility_based(signal, volatility)
        elif method == "risk_parity":
            return self._risk_parity(signal, volatility)
        elif method == "optimal_f":
            return self._optimal_f(signal)
        else:
            return self._fixed_fractional(signal, 0.1)
    
    def _fixed_fractional(
        self,
        signal: Signal,
        fraction: float
    ) -> SizingResult:
        """Fixed fraction of portfolio."""
        value = self.portfolio_value * fraction
        value = min(value, self.portfolio_value * self.max_position_pct)
        
        price = signal.price_target or 0.5
        quantity = value / price if price > 0 else 0
        
        # Calculate risk
        risk = value * (signal.stop_loss / price - 1 if signal.stop_loss else 0.1)
        
        return SizingResult(
            quantity=quantity,
            value=value,
            risk_amount=abs(risk),
            method="fixed",
            confidence=fraction,
            metadata={"fraction": fraction}
        )
    
    def _kelly_criterion(self, signal: Signal) -> SizingResult:
        """Kelly criterion position sizing."""
        # Estimate win probability from confidence
        win_prob = signal.confidence if signal.confidence <= 1 else signal.confidence / 100
        
        # Estimate win/loss ratio from stops
        price = signal.price_target or 0.5
        
        if signal.take_profit and signal.stop_loss:
            win_amount = abs(signal.take_profit - price)
            loss_amount = abs(price - signal.stop_loss)
            win_loss_ratio = win_amount / loss_amount if loss_amount > 0 else 1.5
        else:
            win_loss_ratio = 1.5  # Default
        
        # Kelly formula
        b = win_loss_ratio
        p = win_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Use half Kelly for safety
        kelly_fraction = max(0, kelly_fraction * 0.5)
        
        # Cap at maximum
        kelly_fraction = min(kelly_fraction, self.max_position_pct)
        
        value = self.portfolio_value * kelly_fraction
        quantity = value / price if price > 0 else 0
        risk = value * (1 - signal.stop_loss / price if signal.stop_loss else 0.1)
        
        return SizingResult(
            quantity=quantity,
            value=value,
            risk_amount=abs(risk),
            method="kelly",
            confidence=kelly_fraction,
            metadata={
                "kelly_fraction": kelly_fraction,
                "win_prob": win_prob,
                "win_loss_ratio": win_loss_ratio
            }
        )
    
    def _volatility_based(
        self,
        signal: Signal,
        volatility: float = None
    ) -> SizingResult:
        """Volatility-adjusted position sizing."""
        vol = volatility or 0.2  # Default 20% vol
        price = signal.price_target or 0.5
        
        # Target volatility contribution of 1%
        target_vol_contribution = 0.01
        
        # Position size to achieve target vol contribution
        position_pct = target_vol_contribution / vol if vol > 0 else 0.1
        
        # Cap at maximum
        position_pct = min(position_pct, self.max_position_pct)
        
        value = self.portfolio_value * position_pct
        quantity = value / price if price > 0 else 0
        
        # Risk is related to volatility
        risk = value * vol / np.sqrt(252)  # Daily risk
        
        return SizingResult(
            quantity=quantity,
            value=value,
            risk_amount=risk,
            method="volatility",
            confidence=position_pct,
            metadata={
                "volatility": vol,
                "target_vol_contribution": target_vol_contribution
            }
        )
    
    def _risk_parity(
        self,
        signal: Signal,
        volatility: float = None
    ) -> SizingResult:
        """Risk parity sizing - equal risk contribution."""
        vol = volatility or 0.2
        price = signal.price_target or 0.5
        
        # Target risk per position
        target_risk = self.portfolio_value * self.max_risk_pct
        
        # Size to achieve target risk
        daily_vol = vol / np.sqrt(252)
        value = target_risk / daily_vol if daily_vol > 0 else target_risk
        
        # Cap at maximum
        value = min(value, self.portfolio_value * self.max_position_pct)
        
        quantity = value / price if price > 0 else 0
        
        return SizingResult(
            quantity=quantity,
            value=value,
            risk_amount=target_risk,
            method="risk_parity",
            confidence=value / self.portfolio_value,
            metadata={
                "target_risk": target_risk,
                "volatility": vol
            }
        )
    
    def _optimal_f(self, signal: Signal) -> SizingResult:
        """Optimal f position sizing based on historical results."""
        if len(self._win_history) < 10:
            # Not enough data, use Kelly as fallback
            return self._kelly_criterion(signal)
        
        # Calculate optimal f from historical trades
        all_returns = self._win_history + self._loss_history
        
        # Find f that maximizes geometric growth
        best_f = 0.0
        best_growth = 0.0
        
        for f in np.arange(0.01, 0.5, 0.01):
            growth = 1.0
            for r in all_returns:
                growth *= (1 + f * r)
            
            geometric_mean = growth ** (1 / len(all_returns))
            if geometric_mean > best_growth:
                best_growth = geometric_mean
                best_f = f
        
        # Use half optimal f for safety
        optimal_f = best_f * 0.5
        optimal_f = min(optimal_f, self.max_position_pct)
        
        price = signal.price_target or 0.5
        value = self.portfolio_value * optimal_f
        quantity = value / price if price > 0 else 0
        
        return SizingResult(
            quantity=quantity,
            value=value,
            risk_amount=value * 0.1,  # Estimated
            method="optimal_f",
            confidence=optimal_f,
            metadata={
                "optimal_f": optimal_f,
                "historical_trades": len(all_returns)
            }
        )
    
    def record_result(self, pnl_pct: float, won: bool):
        """Record trade result for adaptive sizing."""
        if won:
            self._win_history.append(pnl_pct)
        else:
            self._loss_history.append(pnl_pct)
        
        # Keep last 100 trades
        self._win_history = self._win_history[-100:]
        self._loss_history = self._loss_history[-100:]
    
    def update_portfolio_value(self, value: float):
        """Update portfolio value."""
        self.portfolio_value = value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sizing statistics."""
        total_trades = len(self._win_history) + len(self._loss_history)
        
        return {
            "portfolio_value": self.portfolio_value,
            "max_position_pct": self.max_position_pct,
            "max_risk_pct": self.max_risk_pct,
            "default_method": self.default_method,
            "historical_trades": total_trades,
            "win_rate": len(self._win_history) / total_trades if total_trades > 0 else 0
        }

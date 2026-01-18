"""Risk management system."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..core.events import Signal, SignalType, Position

logger = logging.getLogger("execution.risk")


class RiskLevel(Enum):
    """Risk levels for dynamic adjustment."""
    MINIMAL = "MINIMAL"      # 25% of normal
    REDUCED = "REDUCED"      # 50% of normal
    NORMAL = "NORMAL"        # 100%
    ELEVATED = "ELEVATED"    # 75% of normal
    MAXIMUM = "MAXIMUM"      # 50% of normal (high vol)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_portfolio_risk: float = 0.02  # 2% max risk per day
    max_position_risk: float = 0.01   # 1% max risk per position
    max_positions: int = 20
    max_correlation: float = 0.7
    max_sector_exposure: float = 0.3  # 30% max in one sector
    max_drawdown: float = 0.10        # 10% max drawdown
    daily_loss_limit: float = 0.03    # 3% daily loss limit
    var_confidence: float = 0.95      # VaR confidence level
    position_size_max: float = 0.15   # 15% max single position


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    portfolio_var: float = 0.0
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    exposure_by_sector: Dict[str, float] = field(default_factory=dict)
    position_count: int = 0
    correlation_risk: float = 0.0
    risk_level: RiskLevel = RiskLevel.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_var": self.portfolio_var,
            "current_drawdown": self.current_drawdown,
            "daily_pnl": self.daily_pnl,
            "exposure_by_sector": self.exposure_by_sector,
            "position_count": self.position_count,
            "correlation_risk": self.correlation_risk,
            "risk_level": self.risk_level.value,
            "timestamp": self.timestamp.isoformat()
        }


class RiskManager:
    """
    Advanced risk management system.
    
    Features:
    - Position sizing with Kelly criterion
    - VaR and CVaR calculation
    - Drawdown protection
    - Correlation-aware exposure limits
    - Dynamic risk level adjustment
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.daily_pnl_history: List[float] = []
        self.portfolio_value_history: List[float] = []
        self.peak_portfolio_value: float = 0
        
        # Risk metrics
        self._metrics = RiskMetrics()
        self._risk_level = RiskLevel.NORMAL
        
        # Correlation matrix
        self._correlations: Dict[Tuple[str, str], float] = {}
        
        # Daily tracking
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._last_reset = datetime.utcnow().date()
    
    def check_signal(
        self,
        signal: Signal,
        portfolio_value: float,
        current_positions: Dict[str, Position] = None
    ) -> Tuple[bool, str, float]:
        """
        Check if signal passes risk checks.
        Returns: (approved, reason, adjusted_size)
        """
        current_positions = current_positions or self.positions
        
        # Reset daily counters if new day
        self._check_daily_reset()
        
        # Check 1: Daily loss limit
        if self._daily_pnl < -self.config.daily_loss_limit * portfolio_value:
            return False, "Daily loss limit reached", 0.0
        
        # Check 2: Position count
        if len(current_positions) >= self.config.max_positions:
            if signal.market_id not in current_positions:
                return False, "Maximum positions reached", 0.0
        
        # Check 3: Drawdown limit
        if self._metrics.current_drawdown > self.config.max_drawdown:
            return False, "Maximum drawdown exceeded", 0.0
        
        # Check 4: Sector exposure
        sector = signal.metadata.get("sector", "other")
        sector_exposure = self._metrics.exposure_by_sector.get(sector, 0)
        if sector_exposure > self.config.max_sector_exposure:
            return False, f"Sector {sector} exposure limit", 0.0
        
        # Check 5: Correlation risk
        correlation_risk = self._calculate_correlation_risk(
            signal.market_id, current_positions
        )
        if correlation_risk > self.config.max_correlation:
            return False, "High correlation with existing positions", 0.0
        
        # Calculate position size
        size = self._calculate_position_size(
            signal, portfolio_value, current_positions
        )
        
        if size <= 0:
            return False, "Position size too small", 0.0
        
        return True, "Approved", size
    
    def _calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_positions: Dict[str, Position]
    ) -> float:
        """Calculate optimal position size."""
        
        # Kelly criterion base
        kelly_size = self._kelly_criterion(signal)
        
        # Risk-adjusted size
        max_risk = self.config.max_position_risk * portfolio_value
        risk_based_size = max_risk / (signal.stop_loss or 0.1) if signal.stop_loss else max_risk
        if kelly_size <= 0:
            kelly_size = risk_based_size
        
        # Volatility adjustment
        vol_adj = self._volatility_adjustment()
        
        # Risk level multiplier
        risk_multiplier = {
            RiskLevel.MINIMAL: 0.25,
            RiskLevel.REDUCED: 0.5,
            RiskLevel.NORMAL: 1.0,
            RiskLevel.ELEVATED: 0.75,
            RiskLevel.MAXIMUM: 0.5
        }.get(self._risk_level, 1.0)
        
        # Combine sizing methods
        base_size = min(kelly_size, risk_based_size) * vol_adj * risk_multiplier
        
        # Apply confidence scaling
        confidence_factor = signal.confidence if signal.confidence <= 1 else signal.confidence / 100
        size = base_size * confidence_factor
        
        # Apply maximum position limit
        max_position = portfolio_value * self.config.position_size_max
        size = min(size, max_position)
        
        return size
    
    def _kelly_criterion(self, signal: Signal) -> float:
        """Calculate Kelly criterion position size."""
        # Estimate win probability from confidence
        win_prob = signal.confidence if signal.confidence <= 1 else signal.confidence / 100
        
        # Estimate win/loss ratio from take profit and stop loss
        if signal.take_profit and signal.stop_loss and signal.price_target:
            win_amount = abs(signal.take_profit - signal.price_target)
            loss_amount = abs(signal.price_target - signal.stop_loss)
            win_loss_ratio = win_amount / loss_amount if loss_amount > 0 else 2
        else:
            win_loss_ratio = 1.5  # Default assumption
        
        # Kelly formula: f = (bp - q) / b
        # where b = win/loss ratio, p = win prob, q = loss prob
        b = win_loss_ratio
        p = win_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Use fractional Kelly (half) for safety
        return max(0, kelly_fraction * 0.5)
    
    def _volatility_adjustment(self) -> float:
        """Adjust for current volatility."""
        if len(self.portfolio_value_history) < 10:
            return 1.0
        
        # Calculate recent volatility
        returns = np.diff(self.portfolio_value_history[-20:]) / self.portfolio_value_history[-20:-1]
        vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.1
        
        # Target volatility of 15%
        target_vol = 0.15
        
        return min(2.0, max(0.3, target_vol / vol)) if vol > 0 else 1.0
    
    def _calculate_correlation_risk(
        self,
        market_id: str,
        positions: Dict[str, Position]
    ) -> float:
        """Calculate correlation risk with existing positions."""
        if not positions:
            return 0.0
        
        max_correlation = 0.0
        for pos_id in positions:
            corr = self._correlations.get((market_id, pos_id), 0)
            corr = max(corr, self._correlations.get((pos_id, market_id), 0))
            max_correlation = max(max_correlation, abs(corr))
        
        return max_correlation
    
    def update_metrics(
        self,
        portfolio_value: float,
        positions: Dict[str, Position] = None
    ) -> RiskMetrics:
        """Update risk metrics."""
        self._check_daily_reset()
        
        positions = positions or self.positions
        
        # Track portfolio value
        self.portfolio_value_history.append(portfolio_value)
        self.portfolio_value_history = self.portfolio_value_history[-500:]
        
        # Update peak
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        # Calculate drawdown
        drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value \
            if self.peak_portfolio_value > 0 else 0
        
        # Calculate VaR
        var = self._calculate_var()
        
        # Calculate sector exposure
        sector_exposure = self._calculate_sector_exposure(positions)
        
        # Determine risk level
        risk_level = self._determine_risk_level(drawdown, var)
        
        self._metrics = RiskMetrics(
            portfolio_var=var,
            current_drawdown=drawdown,
            daily_pnl=self._daily_pnl,
            exposure_by_sector=sector_exposure,
            position_count=len(positions),
            correlation_risk=self._calculate_portfolio_correlation(positions),
            risk_level=risk_level
        )
        
        self._risk_level = risk_level
        
        return self._metrics
    
    def _calculate_var(self, confidence: float = None) -> float:
        """Calculate Value at Risk."""
        confidence = confidence or self.config.var_confidence
        
        if len(self.portfolio_value_history) < 20:
            return 0.05  # Default 5%
        
        returns = np.diff(self.portfolio_value_history[-100:]) / self.portfolio_value_history[-100:-1]
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 10:
            return 0.05
        
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(var)
    
    def _calculate_sector_exposure(
        self,
        positions: Dict[str, Position]
    ) -> Dict[str, float]:
        """Calculate exposure by sector."""
        exposure = {}
        total_value = sum(p.value for p in positions.values())
        
        if total_value == 0:
            return exposure
        
        for pos in positions.values():
            sector = getattr(pos, 'metadata', {}).get('sector', 'other')
            exposure[sector] = exposure.get(sector, 0) + pos.value / total_value
        
        return exposure
    
    def _calculate_portfolio_correlation(
        self,
        positions: Dict[str, Position]
    ) -> float:
        """Calculate average portfolio correlation."""
        if len(positions) < 2:
            return 0.0
        
        correlations = []
        pos_ids = list(positions.keys())
        
        for i, id1 in enumerate(pos_ids):
            for id2 in pos_ids[i+1:]:
                corr = self._correlations.get((id1, id2), 0)
                corr = max(corr, self._correlations.get((id2, id1), 0))
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _determine_risk_level(
        self,
        drawdown: float,
        var: float
    ) -> RiskLevel:
        """Determine current risk level."""
        
        # Check drawdown thresholds
        if drawdown > self.config.max_drawdown * 0.8:
            return RiskLevel.MINIMAL
        elif drawdown > self.config.max_drawdown * 0.5:
            return RiskLevel.REDUCED
        
        # Check VaR
        if var > 0.10:
            return RiskLevel.MAXIMUM
        elif var > 0.05:
            return RiskLevel.ELEVATED
        
        return RiskLevel.NORMAL
    
    def _check_daily_reset(self):
        """Reset daily counters if new day."""
        today = datetime.utcnow().date()
        if today != self._last_reset:
            self.daily_pnl_history.append(self._daily_pnl)
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._last_reset = today
    
    def record_trade_pnl(self, pnl: float):
        """Record PnL from a trade."""
        self._daily_pnl += pnl
        self._daily_trades += 1
    
    def update_correlation(
        self,
        market_a: str,
        market_b: str,
        correlation: float
    ):
        """Update correlation between markets."""
        self._correlations[(market_a, market_b)] = correlation
    
    @property
    def metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        return self._metrics
    
    @property
    def risk_level(self) -> RiskLevel:
        """Get current risk level."""
        return self._risk_level
    
    def get_position_limits(self) -> Dict[str, Any]:
        """Get current position limits based on risk level."""
        multiplier = {
            RiskLevel.MINIMAL: 0.25,
            RiskLevel.REDUCED: 0.5,
            RiskLevel.NORMAL: 1.0,
            RiskLevel.ELEVATED: 0.75,
            RiskLevel.MAXIMUM: 0.5
        }.get(self._risk_level, 1.0)
        
        return {
            "max_positions": int(self.config.max_positions * multiplier),
            "max_position_size": self.config.position_size_max * multiplier,
            "max_sector_exposure": self.config.max_sector_exposure,
            "risk_level": self._risk_level.value,
            "multiplier": multiplier
        }

"""Portfolio management."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core.events import Position, PortfolioSnapshot, Trade

logger = logging.getLogger("execution.portfolio")


@dataclass
class PortfolioConfig:
    """Portfolio configuration."""
    initial_capital: float = 10000.0
    rebalance_threshold: float = 0.1  # 10% drift
    max_positions: int = 20
    target_weights: Dict[str, float] = field(default_factory=dict)


class PortfolioManager:
    """
    Portfolio management system.
    
    Features:
    - Position tracking
    - Performance metrics
    - Rebalancing
    - Historical snapshots
    """
    
    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()
        
        # State
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self._snapshots: List[PortfolioSnapshot] = []
        self._trades: List[Trade] = []
        
        # Performance tracking
        self._peak_value = self.config.initial_capital
        self._daily_returns: List[float] = []
        self._wins = 0
        self._losses = 0
    
    def update_position(
        self,
        market_id: str,
        quantity: float,
        price: float,
        side: str = "BUY",
        strategy_name: Optional[str] = None
    ) -> Optional[float]:
        """Update or create position."""
        realized_pnl = None
        if market_id in self.positions:
            pos = self.positions[market_id]
            if side == "BUY":
                # Add to position
                new_qty = pos.quantity + quantity
                new_avg = (pos.entry_price * pos.quantity + price * quantity) / new_qty
                pos.quantity = new_qty
                pos.entry_price = new_avg
                self.cash -= quantity * price
            else:
                # Reduce position
                if quantity >= pos.quantity:
                    # Close position
                    pnl = (price - pos.entry_price) * pos.quantity
                    self.cash += pos.quantity * price
                    pos.realized_pnl += pnl
                    realized_pnl = pnl
                    
                    if pnl > 0:
                        self._wins += 1
                    else:
                        self._losses += 1
                    
                    del self.positions[market_id]
                else:
                    pos.quantity -= quantity
                    self.cash += quantity * price
                    pnl = (price - pos.entry_price) * quantity
                    pos.realized_pnl += pnl
                    realized_pnl = pnl
        else:
            if side == "BUY":
                self.positions[market_id] = Position(
                    market_id=market_id,
                    side="LONG",
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    strategy_name=strategy_name or ""
                )
                self.cash -= quantity * price
        return realized_pnl
    
    def update_prices(self, prices: Dict[str, float]):
        """Update position prices."""
        for market_id, price in prices.items():
            if market_id in self.positions:
                pos = self.positions[market_id]
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
                pos.updated_at = datetime.utcnow()
    
    def take_snapshot(self) -> PortfolioSnapshot:
        """Take portfolio snapshot."""
        total_value = self.total_value
        positions_value = self.positions_value
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        realized = sum(p.realized_pnl for p in self.positions.values())
        
        # Update peak
        if total_value > self._peak_value:
            self._peak_value = total_value
        
        # Calculate drawdown
        drawdown = (self._peak_value - total_value) / self._peak_value if self._peak_value > 0 else 0
        
        # Calculate metrics
        win_rate = self._wins / (self._wins + self._losses) if (self._wins + self._losses) > 0 else 0
        sharpe = self._calculate_sharpe()
        
        snapshot = PortfolioSnapshot(
            total_value=total_value,
            cash=self.cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            total_pnl=unrealized + realized,
            position_count=len(self.positions),
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=drawdown
        )
        
        self._snapshots.append(snapshot)
        return snapshot
    
    def _calculate_sharpe(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(self._daily_returns) < 10:
            return 0.0
        
        returns = np.array(self._daily_returns)
        excess_returns = returns - risk_free_rate / 252
        
        if np.std(returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def record_daily_return(self, return_pct: float):
        """Record daily return."""
        self._daily_returns.append(return_pct)
        self._daily_returns = self._daily_returns[-365:]  # Keep 1 year
    
    @property
    def total_value(self) -> float:
        """Get total portfolio value."""
        return self.cash + self.positions_value
    
    @property
    def positions_value(self) -> float:
        """Get total positions value."""
        return sum(p.quantity * p.current_price for p in self.positions.values())
    
    @property
    def exposure(self) -> float:
        """Get exposure ratio (positions / total)."""
        if self.total_value == 0:
            return 0.0
        return self.positions_value / self.total_value
    
    def get_allocation(self) -> Dict[str, float]:
        """Get current allocation by position."""
        total = self.total_value
        if total == 0:
            return {}
        
        allocation = {"cash": self.cash / total}
        for market_id, pos in self.positions.items():
            allocation[market_id] = (pos.quantity * pos.current_price) / total
        
        return allocation
    
    def needs_rebalance(self) -> bool:
        """Check if rebalancing is needed."""
        if not self.config.target_weights:
            return False
        
        current = self.get_allocation()
        
        for market_id, target in self.config.target_weights.items():
            current_weight = current.get(market_id, 0)
            if abs(current_weight - target) > self.config.rebalance_threshold:
                return True
        
        return False
    
    def get_rebalance_orders(self) -> List[Dict[str, Any]]:
        """Get orders needed for rebalancing."""
        if not self.config.target_weights:
            return []
        
        orders = []
        total = self.total_value
        current = self.get_allocation()
        
        for market_id, target in self.config.target_weights.items():
            current_weight = current.get(market_id, 0)
            diff = target - current_weight
            
            if abs(diff) > 0.01:  # 1% threshold
                value_change = diff * total
                pos = self.positions.get(market_id)
                price = pos.current_price if pos else 0
                
                if price > 0:
                    quantity = abs(value_change / price)
                    orders.append({
                        "market_id": market_id,
                        "side": "BUY" if diff > 0 else "SELL",
                        "quantity": quantity,
                        "value": abs(value_change)
                    })
        
        return orders
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_pnl = sum(p.realized_pnl + p.unrealized_pnl for p in self.positions.values())
        total_trades = self._wins + self._losses
        
        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "exposure": self.exposure,
            "position_count": len(self.positions),
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / self.config.initial_capital * 100,
            "total_trades": total_trades,
            "wins": self._wins,
            "losses": self._losses,
            "win_rate": self._wins / total_trades * 100 if total_trades > 0 else 0,
            "sharpe_ratio": self._calculate_sharpe(),
            "max_drawdown": (self._peak_value - self.total_value) / self._peak_value if self._peak_value > 0 else 0
        }
    
    def get_positions_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all positions."""
        return [
            {
                "market_id": pos.market_id,
                "side": pos.side,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "value": pos.quantity * pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "pnl_pct": pos.pnl_percent,
                "opened_at": pos.opened_at.isoformat()
            }
            for pos in self.positions.values()
        ]
    
    def export_history(self) -> List[Dict[str, Any]]:
        """Export snapshot history."""
        return [s.to_dict() for s in self._snapshots]

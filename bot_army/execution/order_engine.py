"""Order execution engine."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from ..core.events import Signal, SignalType, Trade, Position, OrderStatus

logger = logging.getLogger("execution.orders")


class ExecutionMode(Enum):
    """Order execution modes."""
    PAPER = "PAPER"       # Simulated trading
    LIVE = "LIVE"         # Real execution
    BACKTEST = "BACKTEST" # Historical simulation


@dataclass
class OrderRequest:
    """Order request."""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal: Signal = None
    side: str = "BUY"
    quantity: float = 0.0
    price: float = 0.0
    order_type: str = "MARKET"
    time_in_force: str = "GTC"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    """Order execution result."""
    order_id: str
    status: OrderStatus
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    fees: float = 0.0
    trade_id: Optional[str] = None
    message: str = ""
    executed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "fees": self.fees,
            "trade_id": self.trade_id,
            "message": self.message,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None
        }


class OrderEngine:
    """
    Order execution engine.
    
    Handles order routing, execution, and management.
    Supports paper trading and live execution modes.
    """
    
    FEE_RATE = 0.001  # 0.1% fee
    
    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PAPER,
        api_client=None
    ):
        self.mode = mode
        self.api_client = api_client
        
        # Order tracking
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.executed_orders: List[OrderResult] = []
        self.positions: Dict[str, Position] = {}
        
        # Paper trading state
        self._paper_cash = 10000.0
        self._paper_positions: Dict[str, Position] = {}
        
        # Performance tracking
        self._total_trades = 0
        self._total_fees = 0.0
        self._slippage_total = 0.0
    
    async def execute_signal(
        self,
        signal: Signal,
        quantity: float
    ) -> OrderResult:
        """Execute a trading signal."""
        
        # Create order request
        order = OrderRequest(
            signal=signal,
            side="BUY" if signal.signal_type == SignalType.BUY else "SELL",
            quantity=quantity,
            price=signal.price_target or 0,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            metadata={
                "strategy": signal.strategy_name,
                "confidence": signal.confidence
            }
        )
        
        # Execute based on mode
        if self.mode == ExecutionMode.PAPER:
            result = await self._execute_paper(order)
        elif self.mode == ExecutionMode.LIVE:
            result = await self._execute_live(order)
        else:
            result = self._execute_backtest(order)
        
        # Track execution
        if result.status == OrderStatus.FILLED:
            self._total_trades += 1
            self._total_fees += result.fees
            self.executed_orders.append(result)
            
            # Update position
            self._update_position(signal, result)
        
        return result
    
    async def _execute_paper(self, order: OrderRequest) -> OrderResult:
        """Execute paper trade."""
        signal = order.signal
        market_id = signal.market_id
        
        # Simulate market price with slippage
        slippage = 0.001  # 0.1% slippage
        if order.side == "BUY":
            fill_price = order.price * (1 + slippage)
        else:
            fill_price = order.price * (1 - slippage)
        
        # Calculate costs
        value = order.quantity * fill_price
        fees = value * self.FEE_RATE
        
        # Check paper balance
        if order.side == "BUY":
            if value + fees > self._paper_cash:
                return OrderResult(
                    order_id=order.order_id,
                    status=OrderStatus.REJECTED,
                    message="Insufficient funds"
                )
            self._paper_cash -= (value + fees)
        else:
            # Allow short sells in paper mode
            self._paper_cash += (value - fees)
        
        # Create trade
        trade_id = str(uuid.uuid4())
        
        # Update paper position
        if order.side == "BUY":
            if market_id in self._paper_positions:
                pos = self._paper_positions[market_id]
                if pos.side == "SHORT":
                    if order.quantity >= pos.quantity:
                        del self._paper_positions[market_id]
                    else:
                        pos.quantity -= order.quantity
                else:
                    new_qty = pos.quantity + order.quantity
                    new_avg = (pos.entry_price * pos.quantity + fill_price * order.quantity) / new_qty
                    pos.quantity = new_qty
                    pos.entry_price = new_avg
            else:
                self._paper_positions[market_id] = Position(
                    market_id=market_id,
                    side="LONG",
                    quantity=order.quantity,
                    entry_price=fill_price,
                    current_price=fill_price,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    strategy_name=signal.strategy_name
                )
        else:
            pos = self._paper_positions.get(market_id)
            if pos:
                if pos.side == "LONG":
                    if order.quantity >= pos.quantity:
                        del self._paper_positions[market_id]
                    else:
                        pos.quantity -= order.quantity
                else:
                    new_qty = pos.quantity + order.quantity
                    new_avg = (pos.entry_price * pos.quantity + fill_price * order.quantity) / new_qty
                    pos.quantity = new_qty
                    pos.entry_price = new_avg
            else:
                self._paper_positions[market_id] = Position(
                    market_id=market_id,
                    side="SHORT",
                    quantity=order.quantity,
                    entry_price=fill_price,
                    current_price=fill_price,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    strategy_name=signal.strategy_name
                )
        
        self._slippage_total += abs(fill_price - order.price)
        
        logger.info(f"Paper {order.side}: {order.quantity:.2f} @ {fill_price:.4f}")
        
        return OrderResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            filled_price=fill_price,
            fees=fees,
            trade_id=trade_id,
            message="Paper trade executed",
            executed_at=datetime.utcnow()
        )
    
    async def _execute_live(self, order: OrderRequest) -> OrderResult:
        """Execute live trade via API."""
        if not self.api_client:
            return OrderResult(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                message="No API client configured"
            )
        
        try:
            # Call Polymarket API
            # This would integrate with actual Polymarket CLOB API
            response = await self.api_client.place_order(
                market_id=order.signal.market_id,
                side=order.side,
                size=order.quantity,
                price=order.price
            )
            
            if response.get("success"):
                return OrderResult(
                    order_id=order.order_id,
                    status=OrderStatus.FILLED,
                    filled_quantity=response.get("filled_size", order.quantity),
                    filled_price=response.get("avg_price", order.price),
                    fees=response.get("fees", 0),
                    trade_id=response.get("trade_id"),
                    message="Order filled",
                    executed_at=datetime.utcnow()
                )
            else:
                return OrderResult(
                    order_id=order.order_id,
                    status=OrderStatus.REJECTED,
                    message=response.get("error", "Unknown error")
                )
                
        except Exception as e:
            logger.error(f"Live execution error: {e}")
            return OrderResult(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                message=str(e)
            )
    
    def _execute_backtest(self, order: OrderRequest) -> OrderResult:
        """Execute backtest trade."""
        # Assume perfect fill at target price
        return OrderResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            filled_price=order.price,
            fees=order.quantity * order.price * self.FEE_RATE,
            trade_id=str(uuid.uuid4()),
            message="Backtest fill",
            executed_at=datetime.utcnow()
        )
    
    def _update_position(self, signal: Signal, result: OrderResult):
        """Update position after execution."""
        market_id = signal.market_id
        
        if signal.signal_type == SignalType.BUY:
            if market_id in self.positions:
                pos = self.positions[market_id]
                if pos.side == "SHORT":
                    if result.filled_quantity >= pos.quantity:
                        del self.positions[market_id]
                    else:
                        pos.quantity -= result.filled_quantity
                else:
                    new_qty = pos.quantity + result.filled_quantity
                    new_avg = (pos.entry_price * pos.quantity +
                               result.filled_price * result.filled_quantity) / new_qty
                    pos.quantity = new_qty
                    pos.entry_price = new_avg
            else:
                self.positions[market_id] = Position(
                    market_id=market_id,
                    side="LONG",
                    quantity=result.filled_quantity,
                    entry_price=result.filled_price,
                    current_price=result.filled_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    strategy_name=signal.strategy_name
                )
        else:
            # Close or reduce position, or open short
            if market_id in self.positions:
                pos = self.positions[market_id]
                if pos.side == "LONG":
                    pnl = (result.filled_price - pos.entry_price) * result.filled_quantity
                    pos.realized_pnl += pnl
                    
                    if result.filled_quantity >= pos.quantity:
                        del self.positions[market_id]
                    else:
                        pos.quantity -= result.filled_quantity
                else:
                    new_qty = pos.quantity + result.filled_quantity
                    new_avg = (pos.entry_price * pos.quantity +
                               result.filled_price * result.filled_quantity) / new_qty
                    pos.quantity = new_qty
                    pos.entry_price = new_avg
            else:
                self.positions[market_id] = Position(
                    market_id=market_id,
                    side="SHORT",
                    quantity=result.filled_quantity,
                    entry_price=result.filled_price,
                    current_price=result.filled_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    strategy_name=signal.strategy_name
                )
    
    async def close_position(
        self,
        market_id: str,
        current_price: float
    ) -> Optional[OrderResult]:
        """Close an existing position."""
        pos = self.positions.get(market_id) or self._paper_positions.get(market_id)
        
        if not pos:
            return None
        
        signal = Signal(
            market_id=market_id,
            signal_type=SignalType.SELL,
            confidence=1.0,
            price_target=current_price
        )
        
        return await self.execute_signal(signal, pos.quantity)
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        for market_id, price in prices.items():
            if market_id in self.positions:
                pos = self.positions[market_id]
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
            
            if market_id in self._paper_positions:
                pos = self._paper_positions[market_id]
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
    
    def check_stops(self, prices: Dict[str, float]) -> List[str]:
        """Check stop loss and take profit for all positions."""
        positions_to_close = []
        
        for market_id, pos in {**self.positions, **self._paper_positions}.items():
            if market_id not in prices:
                continue
            
            current = prices[market_id]
            
            # Stop loss
            if pos.stop_loss:
                if pos.side == "SHORT" and current >= pos.stop_loss:
                    positions_to_close.append(market_id)
                    logger.info(f"Stop loss triggered for short {market_id}")
                elif pos.side != "SHORT" and current <= pos.stop_loss:
                    positions_to_close.append(market_id)
                    logger.info(f"Stop loss triggered for {market_id}")
            
            # Take profit
            elif pos.take_profit:
                if pos.side == "SHORT" and current <= pos.take_profit:
                    positions_to_close.append(market_id)
                    logger.info(f"Take profit triggered for short {market_id}")
                elif pos.side != "SHORT" and current >= pos.take_profit:
                    positions_to_close.append(market_id)
                    logger.info(f"Take profit triggered for {market_id}")
        
        return positions_to_close
    
    @property
    def paper_portfolio_value(self) -> float:
        """Get paper portfolio value."""
        positions_value = sum(
            p.quantity * p.current_price
            for p in self._paper_positions.values()
        )
        return self._paper_cash + positions_value
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "mode": self.mode.value,
            "total_trades": self._total_trades,
            "total_fees": self._total_fees,
            "avg_slippage": self._slippage_total / self._total_trades if self._total_trades > 0 else 0,
            "open_positions": len(self.positions) + len(self._paper_positions),
            "pending_orders": len(self.pending_orders),
            "paper_cash": self._paper_cash,
            "paper_portfolio_value": self.paper_portfolio_value
        }

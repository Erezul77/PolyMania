"""Event and data classes for the trading system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid


class SignalType(Enum):
    """Signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order statuses."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Event:
    """Polymarket event data."""
    event_id: str
    title: str
    slug: str
    category: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    volume: float = 0.0
    liquidity: float = 0.0
    outcomes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    collected_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "slug": self.slug,
            "category": self.category,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "volume": self.volume,
            "liquidity": self.liquidity,
            "outcomes": self.outcomes,
            "metadata": self.metadata,
            "collected_at": self.collected_at.isoformat()
        }


@dataclass
class MarketData:
    """Market tick data."""
    market_id: str
    event_id: str
    outcome: str
    price: float
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume_24h: float = 0.0
    volume_1h: float = 0.0
    open_interest: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "event_id": self.event_id,
            "outcome": self.outcome,
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "volume_24h": self.volume_24h,
            "volume_1h": self.volume_1h,
            "open_interest": self.open_interest,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Signal:
    """Trading signal."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    market_id: str = ""
    event_id: str = ""
    signal_type: SignalType = SignalType.HOLD
    confidence: float = 0.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = ""
    features: Dict[str, float] = field(default_factory=dict)
    ml_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "market_id": self.market_id,
            "event_id": self.event_id,
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy_name": self.strategy_name,
            "features": self.features,
            "ml_scores": self.ml_scores,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata
        }


@dataclass 
class Trade:
    """Executed trade."""
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_id: str = ""
    market_id: str = ""
    event_id: str = ""
    side: str = "BUY"
    quantity: float = 0.0
    price: float = 0.0
    value: float = 0.0
    fees: float = 0.0
    pnl: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    strategy_name: str = ""
    executed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "signal_id": self.signal_id,
            "market_id": self.market_id,
            "event_id": self.event_id,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "value": self.value,
            "fees": self.fees,
            "pnl": self.pnl,
            "status": self.status.value,
            "strategy_name": self.strategy_name,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class Position:
    """Open position."""
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    market_id: str = ""
    event_id: str = ""
    outcome: str = ""
    side: str = "LONG"
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = ""
    opened_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def pnl_percent(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "market_id": self.market_id,
            "event_id": self.event_id,
            "outcome": self.outcome,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "value": self.value,
            "pnl_percent": self.pnl_percent,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy_name": self.strategy_name,
            "opened_at": self.opened_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class PortfolioSnapshot:
    """Portfolio state snapshot."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    total_value: float = 0.0
    cash: float = 0.0
    positions_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    position_count: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "total_value": self.total_value,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": self.total_pnl,
            "position_count": self.position_count,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "timestamp": self.timestamp.isoformat()
        }

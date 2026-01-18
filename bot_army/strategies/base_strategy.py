"""Base strategy class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..core.events import Signal, SignalType

logger = logging.getLogger("strategy.base")


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str
    enabled: bool = True
    weight: float = 1.0
    min_confidence: float = 0.5
    max_positions: int = 10
    position_size_pct: float = 0.1
    stop_loss_pct: float = 0.1
    take_profit_pct: float = 0.2
    cooldown_seconds: int = 300
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Result from strategy evaluation."""
    signal: Signal
    strategy_name: str
    score: float
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal.to_dict(),
            "strategy_name": self.strategy_name,
            "score": self.score,
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    Provides common infrastructure and interface.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger(f"strategy.{config.name}")
        
        # Performance tracking
        self._signals_generated = 0
        self._signals_profitable = 0
        self._total_pnl = 0.0
        self._last_signal_time: Dict[str, datetime] = {}
        
        # State
        self._active = True
        self._positions: Dict[str, Any] = {}
    
    @abstractmethod
    def analyze(
        self,
        market_id: str,
        features: Dict[str, float],
        price_history: List[float],
        orderbook: Dict = None,
        trades: List[Dict] = None
    ) -> Optional[StrategyResult]:
        """
        Analyze market and generate signal.
        Returns None if no signal.
        """
        pass
    
    def should_generate_signal(self, market_id: str) -> bool:
        """Check if enough time has passed since last signal."""
        if market_id not in self._last_signal_time:
            return True
        
        elapsed = (datetime.utcnow() - self._last_signal_time[market_id]).total_seconds()
        return elapsed >= self.config.cooldown_seconds
    
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_positions: int
    ) -> float:
        """Calculate position size for signal."""
        if current_positions >= self.config.max_positions:
            return 0.0
        
        # Base size
        base_size = portfolio_value * self.config.position_size_pct
        
        # Adjust by confidence
        confidence_adj = signal.confidence / 100 if signal.confidence > 1 else signal.confidence
        
        # Adjust by remaining capacity
        capacity_adj = 1.0 - (current_positions / self.config.max_positions)
        
        return base_size * confidence_adj * capacity_adj
    
    def calculate_stops(
        self,
        entry_price: float,
        signal_type: SignalType
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit prices."""
        if signal_type == SignalType.BUY:
            stop_loss = entry_price * (1 - self.config.stop_loss_pct)
            take_profit = entry_price * (1 + self.config.take_profit_pct)
        else:  # SELL
            stop_loss = entry_price * (1 + self.config.stop_loss_pct)
            take_profit = entry_price * (1 - self.config.take_profit_pct)
        
        return stop_loss, take_profit
    
    def create_signal(
        self,
        market_id: str,
        signal_type: SignalType,
        confidence: float,
        price: float,
        features: Dict[str, float] = None,
        **kwargs
    ) -> Signal:
        """Create a trading signal."""
        stop_loss, take_profit = self.calculate_stops(price, signal_type)
        
        signal = Signal(
            market_id=market_id,
            signal_type=signal_type,
            confidence=confidence,
            price_target=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=self.config.name,
            features=features or {},
            metadata=kwargs
        )
        
        self._signals_generated += 1
        self._last_signal_time[market_id] = datetime.utcnow()
        
        return signal
    
    def update_performance(
        self,
        signal_id: str,
        pnl: float,
        profitable: bool
    ):
        """Update strategy performance metrics."""
        self._total_pnl += pnl
        if profitable:
            self._signals_profitable += 1
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        win_rate = (
            self._signals_profitable / self._signals_generated * 100
            if self._signals_generated > 0 else 0
        )
        
        return {
            "name": self.config.name,
            "enabled": self.config.enabled,
            "signals_generated": self._signals_generated,
            "signals_profitable": self._signals_profitable,
            "win_rate": win_rate,
            "total_pnl": self._total_pnl,
            "active_positions": len(self._positions)
        }
    
    def validate_signal(self, signal: Signal) -> bool:
        """Validate signal meets minimum criteria."""
        if signal.confidence < self.config.min_confidence:
            return False
        
        if signal.signal_type == SignalType.HOLD:
            return False
        
        return True

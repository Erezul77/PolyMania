"""Base classes for the Bot Army system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging


@dataclass
class BotConfig:
    """Configuration for a bot instance."""
    name: str
    enabled: bool = True
    interval_seconds: float = 1.0
    max_retries: int = 3
    params: Dict[str, Any] = field(default_factory=dict)


class BaseBot(ABC):
    """Base class for all bots in the army."""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = logging.getLogger(f"bot.{config.name}")
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_run: Optional[datetime] = None
        self._error_count = 0
        self._success_count = 0
    
    @abstractmethod
    async def execute(self) -> Any:
        """Execute the bot's main logic."""
        pass
    
    async def start(self):
        """Start the bot."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        self.logger.info(f"Bot {self.config.name} started")
    
    async def stop(self):
        """Stop the bot."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info(f"Bot {self.config.name} stopped")
    
    async def _run_loop(self):
        """Main run loop."""
        while self._running:
            try:
                await self.execute()
                self._success_count += 1
                self._last_run = datetime.utcnow()
                self._error_count = 0
            except Exception as e:
                self._error_count += 1
                self.logger.error(f"Error in {self.config.name}: {e}")
                if self._error_count >= self.config.max_retries:
                    self.logger.critical(f"Max retries reached for {self.config.name}")
                    await asyncio.sleep(60)  # Backoff
            
            await asyncio.sleep(self.config.interval_seconds)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get bot statistics."""
        return {
            "name": self.config.name,
            "running": self._running,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "success_count": self._success_count,
            "error_count": self._error_count
        }


class BaseCollector(BaseBot):
    """Base class for data collectors."""
    
    def __init__(self, config: BotConfig, db_manager=None, cache_manager=None):
        super().__init__(config)
        self.db = db_manager
        self.cache = cache_manager
        self._collected_count = 0
    
    @abstractmethod
    async def collect(self) -> List[Any]:
        """Collect data from source."""
        pass
    
    async def execute(self) -> Any:
        """Execute collection and storage."""
        data = await self.collect()
        if data:
            self._collected_count += len(data)
            if self.db:
                await self.db.store_batch(data)
            if self.cache:
                # Convert to dicts for JSON serialization
                json_data = [d.to_dict() if hasattr(d, 'to_dict') else d for d in data]
                await self.cache.publish(self.config.name, json_data)
        return data
    
    @property
    def stats(self) -> Dict[str, Any]:
        stats = super().stats
        stats["collected_count"] = self._collected_count
        return stats


class BaseStrategy(ABC):
    """Base class for trading strategies."""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.logger = logging.getLogger(f"strategy.{name}")
        self._signals_generated = 0
        self._trades_executed = 0
        self._pnl = 0.0
    
    @abstractmethod
    async def generate_signals(self, data: Any) -> List['Signal']:
        """Generate trading signals from data."""
        pass
    
    @abstractmethod
    async def calculate_position_size(self, signal: 'Signal', portfolio: Any) -> float:
        """Calculate position size for a signal."""
        pass
    
    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "signals_generated": self._signals_generated,
            "trades_executed": self._trades_executed,
            "pnl": self._pnl
        }


class Signal:
    """Trading signal placeholder - imported from events."""
    pass

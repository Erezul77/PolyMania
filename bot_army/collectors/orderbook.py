"""Orderbook collector and analyzer."""

import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..core.base import BaseCollector, BotConfig

logger = logging.getLogger("collector.orderbook")


@dataclass
class OrderbookLevel:
    """Single level in orderbook."""
    price: float
    size: float
    count: int = 1


@dataclass
class Orderbook:
    """Full orderbook snapshot."""
    market_id: str
    token_id: str
    bids: List[OrderbookLevel] = field(default_factory=list)
    asks: List[OrderbookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 1.0
    
    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        if self.mid_price == 0:
            return 0
        return (self.spread / self.mid_price) * 10000
    
    @property
    def bid_depth(self) -> float:
        """Total bid liquidity."""
        return sum(level.size for level in self.bids)
    
    @property
    def ask_depth(self) -> float:
        """Total ask liquidity."""
        return sum(level.size for level in self.asks)
    
    @property
    def imbalance(self) -> float:
        """Order book imbalance (-1 to 1)."""
        total = self.bid_depth + self.ask_depth
        if total == 0:
            return 0
        return (self.bid_depth - self.ask_depth) / total
    
    def depth_at_price(self, side: str, price_levels: int = 5) -> float:
        """Get depth at first N price levels."""
        levels = self.bids[:price_levels] if side == "bid" else self.asks[:price_levels]
        return sum(level.size for level in levels)
    
    def vwap(self, side: str, size: float) -> float:
        """Volume-weighted average price to fill size."""
        levels = self.bids if side == "bid" else self.asks
        remaining = size
        total_value = 0.0
        
        for level in levels:
            fill = min(remaining, level.size)
            total_value += fill * level.price
            remaining -= fill
            if remaining <= 0:
                break
        
        filled = size - remaining
        return total_value / filled if filled > 0 else 0
    
    def slippage(self, side: str, size: float) -> float:
        """Calculate slippage for given size."""
        if side == "bid":
            return self.best_bid - self.vwap("bid", size)
        else:
            return self.vwap("ask", size) - self.best_ask
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "token_id": self.token_id,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "imbalance": self.imbalance,
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "timestamp": self.timestamp.isoformat()
        }


class OrderbookCollector(BaseCollector):
    """Collects and analyzes orderbook data."""
    
    CLOB_API = "https://clob.polymarket.com"
    
    def __init__(
        self,
        config: BotConfig,
        token_ids: List[str] = None,
        db_manager=None,
        cache_manager=None
    ):
        super().__init__(config, db_manager, cache_manager)
        self.token_ids = token_ids or []
        self._session: Optional[aiohttp.ClientSession] = None
        self.orderbooks: Dict[str, Orderbook] = {}
        self._history: Dict[str, List[Orderbook]] = {}
        self._history_size = 100
    
    async def start(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        await super().start()
    
    async def stop(self):
        await super().stop()
        if self._session:
            await self._session.close()
    
    async def collect(self) -> List[Orderbook]:
        """Collect orderbooks for all tracked tokens."""
        orderbooks = []
        
        for token_id in self.token_ids:
            try:
                ob = await self._fetch_orderbook(token_id)
                if ob:
                    orderbooks.append(ob)
                    self.orderbooks[token_id] = ob
                    
                    # Store history
                    if token_id not in self._history:
                        self._history[token_id] = []
                    self._history[token_id].append(ob)
                    self._history[token_id] = self._history[token_id][-self._history_size:]
            except Exception as e:
                logger.error(f"Error fetching orderbook {token_id}: {e}")
        
        # Publish to cache
        if self.cache and orderbooks:
            await self.cache.publish(
                "orderbooks",
                [ob.to_dict() for ob in orderbooks]
            )
        
        return orderbooks
    
    async def _fetch_orderbook(self, token_id: str) -> Optional[Orderbook]:
        """Fetch orderbook from CLOB API."""
        try:
            url = f"{self.CLOB_API}/book"
            params = {"token_id": token_id}
            
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_orderbook(token_id, data)
                return None
        except Exception as e:
            logger.error(f"Fetch orderbook error: {e}")
            return None
    
    def _parse_orderbook(self, token_id: str, data: Dict) -> Orderbook:
        """Parse API response to Orderbook."""
        bids = [
            OrderbookLevel(
                price=float(level.get("price", 0)),
                size=float(level.get("size", 0))
            )
            for level in data.get("bids", [])
        ]
        
        asks = [
            OrderbookLevel(
                price=float(level.get("price", 0)),
                size=float(level.get("size", 0))
            )
            for level in data.get("asks", [])
        ]
        
        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        
        return Orderbook(
            market_id=data.get("market", ""),
            token_id=token_id,
            bids=bids,
            asks=asks
        )
    
    def add_token(self, token_id: str):
        """Add token to tracking list."""
        if token_id not in self.token_ids:
            self.token_ids.append(token_id)
    
    def remove_token(self, token_id: str):
        """Remove token from tracking list."""
        if token_id in self.token_ids:
            self.token_ids.remove(token_id)
    
    def get_orderbook(self, token_id: str) -> Optional[Orderbook]:
        """Get latest orderbook for token."""
        return self.orderbooks.get(token_id)
    
    def get_history(self, token_id: str) -> List[Orderbook]:
        """Get orderbook history for token."""
        return self._history.get(token_id, [])
    
    def analyze_flow(self, token_id: str) -> Dict[str, Any]:
        """Analyze order flow from history."""
        history = self.get_history(token_id)
        if len(history) < 2:
            return {}
        
        # Calculate changes
        prev, curr = history[-2], history[-1]
        
        bid_delta = curr.bid_depth - prev.bid_depth
        ask_delta = curr.ask_depth - prev.ask_depth
        imbalance_delta = curr.imbalance - prev.imbalance
        spread_delta = curr.spread - prev.spread
        
        return {
            "bid_delta": bid_delta,
            "ask_delta": ask_delta,
            "net_flow": bid_delta - ask_delta,
            "imbalance_delta": imbalance_delta,
            "spread_delta": spread_delta,
            "pressure": "buy" if bid_delta > ask_delta else "sell",
            "timestamp": curr.timestamp.isoformat()
        }

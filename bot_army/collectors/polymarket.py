"""Polymarket API collector."""

import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.base import BaseCollector, BotConfig
from ..core.events import Event, MarketData

logger = logging.getLogger("collector.polymarket")


class PolymarketCollector(BaseCollector):
    """Collects live data from Polymarket APIs."""
    
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
    DATA_API = "https://data-api.polymarket.com"
    
    def __init__(
        self,
        config: BotConfig,
        db_manager=None,
        cache_manager=None
    ):
        super().__init__(config, db_manager, cache_manager)
        self._session: Optional[aiohttp.ClientSession] = None
        self.markets_cache: Dict[str, MarketData] = {}
    
    async def start(self):
        """Start collector with HTTP session."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        await super().start()
    
    async def stop(self):
        """Stop collector and close session."""
        await super().stop()
        if self._session:
            await self._session.close()
    
    async def collect(self) -> List[MarketData]:
        """Collect market data from Polymarket."""
        try:
            # Fetch active markets
            markets = await self._fetch_markets()
            
            # Convert to MarketData objects
            market_data = []
            for market in markets:
                md = self._parse_market(market)
                if md:
                    market_data.append(md)
                    self.markets_cache[md.market_id] = md
            
            logger.info(f"Collected {len(market_data)} markets")
            return market_data
            
        except Exception as e:
            logger.error(f"Collection error: {e}")
            return []
    
    async def _fetch_markets(self, limit: int = 100) -> List[Dict]:
        """Fetch active markets from Gamma API."""
        try:
            url = f"{self.GAMMA_API}/markets"
            params = {
                "limit": str(limit),
                "active": "true",
                "closed": "false"
            }
            
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"API returned {resp.status}")
                    return []
        except Exception as e:
            logger.error(f"Fetch markets error: {e}")
            return []
    
    async def fetch_events(self, limit: int = 50) -> List[Event]:
        """Fetch events from Gamma API."""
        try:
            url = f"{self.GAMMA_API}/events"
            params = {"limit": str(limit), "active": "true"}
            
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [self._parse_event(e) for e in data if e]
                return []
        except Exception as e:
            logger.error(f"Fetch events error: {e}")
            return []
    
    async def fetch_orderbook(self, token_id: str) -> Dict:
        """Fetch orderbook for a token."""
        try:
            url = f"{self.CLOB_API}/book"
            params = {"token_id": token_id}
            
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}
        except Exception as e:
            logger.error(f"Fetch orderbook error: {e}")
            return {}
    
    async def fetch_trades(
        self, 
        market_id: str = None, 
        limit: int = 100
    ) -> List[Dict]:
        """Fetch recent trades."""
        try:
            url = f"{self.DATA_API}/trades"
            params = {"limit": limit}
            if market_id:
                params["market"] = market_id
            
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []
        except Exception as e:
            logger.error(f"Fetch trades error: {e}")
            return []
    
    async def fetch_price_history(
        self,
        market_id: str,
        interval: str = "1h",
        limit: int = 100
    ) -> List[Dict]:
        """Fetch OHLCV price history."""
        try:
            url = f"{self.CLOB_API}/prices-history"
            params = {
                "market": market_id,
                "interval": interval,
                "limit": limit
            }
            
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []
        except Exception as e:
            logger.error(f"Fetch price history error: {e}")
            return []
    
    def _parse_market(self, data: Dict) -> Optional[MarketData]:
        """Parse API response to MarketData."""
        try:
            return MarketData(
                market_id=data.get("id") or data.get("conditionId", ""),
                event_id=data.get("eventId", ""),
                outcome=data.get("outcome", data.get("question", "")),
                price=float(data.get("outcomePrices", [0.5])[0] if isinstance(data.get("outcomePrices"), list) else data.get("lastTradePrice", 0.5)),
                bid=float(data.get("bestBid", 0)),
                ask=float(data.get("bestAsk", 0)),
                spread=float(data.get("spread", 0)),
                volume_24h=float(data.get("volume24hr", data.get("volume", 0))),
                volume_1h=float(data.get("volume1hr", 0)),
                open_interest=float(data.get("openInterest", data.get("liquidity", 0))),
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.debug(f"Parse market error: {e}")
            return None
    
    def _parse_event(self, data: Dict) -> Optional[Event]:
        """Parse API response to Event."""
        try:
            return Event(
                event_id=data.get("id", ""),
                title=data.get("title", ""),
                slug=data.get("slug", ""),
                category=data.get("category", "other"),
                start_date=datetime.fromisoformat(data["startDate"].replace("Z", "")) if data.get("startDate") else None,
                end_date=datetime.fromisoformat(data["endDate"].replace("Z", "")) if data.get("endDate") else None,
                volume=float(data.get("volume", 0)),
                liquidity=float(data.get("liquidity", 0)),
                outcomes=data.get("markets", []),
                metadata=data
            )
        except Exception as e:
            logger.debug(f"Parse event error: {e}")
            return None


class HighFrequencyCollector(PolymarketCollector):
    """High-frequency data collector for tick data."""
    
    def __init__(self, config: BotConfig, db_manager=None, cache_manager=None):
        config.interval_seconds = 0.5  # 500ms polling
        super().__init__(config, db_manager, cache_manager)
        self._tick_buffer: List[MarketData] = []
        self._buffer_size = 100
    
    async def collect(self) -> List[MarketData]:
        """Collect tick-level data."""
        data = await super().collect()
        
        # Buffer ticks for batch processing
        self._tick_buffer.extend(data)
        
        if len(self._tick_buffer) >= self._buffer_size:
            # Flush buffer
            batch = self._tick_buffer[:]
            self._tick_buffer = []
            
            # Publish to Redis for real-time consumption
            if self.cache:
                await self.cache.publish("ticks", [d.to_dict() for d in batch])
            
            return batch
        
        return []

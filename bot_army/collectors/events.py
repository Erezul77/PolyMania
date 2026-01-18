"""Events collector for Polymarket."""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import defaultdict

from ..core.base import BaseCollector, BotConfig
from ..core.events import Event

logger = logging.getLogger("collector.events")


class EventsCollector(BaseCollector):
    """Collects and tracks Polymarket events."""
    
    GAMMA_API = "https://gamma-api.polymarket.com"
    
    # Event categories for segment routing
    CATEGORIES = {
        "crypto": ["bitcoin", "ethereum", "crypto", "btc", "eth", "solana", "defi"],
        "politics": ["election", "president", "congress", "senate", "vote", "government", "trump", "biden"],
        "sports": ["nfl", "nba", "mlb", "soccer", "football", "championship", "game", "match"],
        "entertainment": ["oscar", "grammy", "movie", "film", "celebrity", "award"],
        "science": ["space", "nasa", "climate", "discovery", "research"],
        "economics": ["fed", "interest", "inflation", "gdp", "unemployment", "economy"],
        "weather": ["hurricane", "earthquake", "temperature", "storm"],
        "tech": ["apple", "google", "ai", "artificial intelligence", "openai", "tesla"]
    }
    
    def __init__(
        self,
        config: BotConfig,
        db_manager=None,
        cache_manager=None
    ):
        super().__init__(config, db_manager, cache_manager)
        self._session: Optional[aiohttp.ClientSession] = None
        self.events: Dict[str, Event] = {}
        self._events_by_category: Dict[str, List[Event]] = defaultdict(list)
        self._hot_events: List[Event] = []
    
    async def start(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        await super().start()
    
    async def stop(self):
        await super().stop()
        if self._session:
            await self._session.close()
    
    async def collect(self) -> List[Event]:
        """Collect active events."""
        try:
            events_data = await self._fetch_events()
            events = []
            
            for data in events_data:
                event = self._parse_event(data)
                if event:
                    events.append(event)
                    self.events[event.event_id] = event
                    
                    # Categorize
                    category = self._detect_category(event)
                    self._events_by_category[category].append(event)
            
            # Update hot events (high volume)
            self._hot_events = sorted(
                events, 
                key=lambda e: e.volume, 
                reverse=True
            )[:20]
            
            # Publish to cache
            if self.cache and events:
                await self.cache.set(
                    "events:active",
                    [e.to_dict() for e in events],
                    ttl=300
                )
                await self.cache.set(
                    "events:hot",
                    [e.to_dict() for e in self._hot_events],
                    ttl=60
                )
            
            logger.info(f"Collected {len(events)} events")
            return events
            
        except Exception as e:
            logger.error(f"Event collection error: {e}")
            return []
    
    async def _fetch_events(self, limit: int = 100) -> List[Dict]:
        """Fetch events from Gamma API."""
        try:
            url = f"{self.GAMMA_API}/events"
            params = {"limit": str(limit), "active": "true"}
            
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []
        except Exception as e:
            logger.error(f"Fetch events error: {e}")
            return []
    
    async def fetch_event(self, event_id: str) -> Optional[Event]:
        """Fetch specific event."""
        try:
            url = f"{self.GAMMA_API}/events/{event_id}"
            
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_event(data)
                return None
        except Exception as e:
            logger.error(f"Fetch event error: {e}")
            return None
    
    def _parse_event(self, data: Dict) -> Optional[Event]:
        """Parse API response to Event."""
        try:
            return Event(
                event_id=data.get("id", ""),
                title=data.get("title", ""),
                slug=data.get("slug", ""),
                category=data.get("category", "other"),
                start_date=datetime.fromisoformat(
                    data["startDate"].replace("Z", "")
                ) if data.get("startDate") else None,
                end_date=datetime.fromisoformat(
                    data["endDate"].replace("Z", "")
                ) if data.get("endDate") else None,
                volume=float(data.get("volume", 0)),
                liquidity=float(data.get("liquidity", 0)),
                outcomes=data.get("markets", []),
                metadata=data
            )
        except Exception as e:
            logger.debug(f"Parse event error: {e}")
            return None
    
    def _detect_category(self, event: Event) -> str:
        """Detect event category from title and metadata."""
        title_lower = event.title.lower()
        
        for category, keywords in self.CATEGORIES.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return category
        
        return event.category or "other"
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get cached event."""
        return self.events.get(event_id)
    
    def get_events_by_category(self, category: str) -> List[Event]:
        """Get events in category."""
        return self._events_by_category.get(category, [])
    
    def get_hot_events(self, limit: int = 10) -> List[Event]:
        """Get highest volume events."""
        return self._hot_events[:limit]
    
    def get_ending_soon(self, hours: int = 24) -> List[Event]:
        """Get events ending soon."""
        cutoff = datetime.utcnow() + timedelta(hours=hours)
        
        return [
            e for e in self.events.values()
            if e.end_date and e.end_date < cutoff and e.end_date > datetime.utcnow()
        ]
    
    def analyze_event(self, event_id: str) -> Dict[str, Any]:
        """Analyze event characteristics."""
        event = self.get_event(event_id)
        if not event:
            return {}
        
        now = datetime.utcnow()
        
        # Time analysis
        time_to_end = None
        if event.end_date:
            time_to_end = (event.end_date - now).total_seconds() / 3600
        
        event_duration = None
        if event.start_date and event.end_date:
            event_duration = (event.end_date - event.start_date).total_seconds() / 3600
        
        progress = None
        if event.start_date and event.end_date and event.start_date < now:
            elapsed = (now - event.start_date).total_seconds()
            total = (event.end_date - event.start_date).total_seconds()
            progress = elapsed / total if total > 0 else 0
        
        return {
            "event_id": event_id,
            "category": self._detect_category(event),
            "volume": event.volume,
            "liquidity": event.liquidity,
            "outcome_count": len(event.outcomes),
            "time_to_end_hours": time_to_end,
            "event_duration_hours": event_duration,
            "progress": progress,
            "is_ending_soon": time_to_end is not None and time_to_end < 24,
            "is_high_volume": event.volume > 100000,
            "is_liquid": event.liquidity > 50000
        }

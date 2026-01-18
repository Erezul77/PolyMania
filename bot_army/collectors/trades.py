"""Trade data collector and analyzer."""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.base import BaseCollector, BotConfig

logger = logging.getLogger("collector.trades")


@dataclass
class TradeRecord:
    """Individual trade record."""
    trade_id: str
    market_id: str
    outcome: str
    side: str  # BUY or SELL
    price: float
    size: float
    value: float
    maker: str = ""
    taker: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "market_id": self.market_id,
            "outcome": self.outcome,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "value": self.value,
            "maker": self.maker,
            "taker": self.taker,
            "timestamp": self.timestamp.isoformat()
        }


class TradesCollector(BaseCollector):
    """Collects and analyzes trade data."""
    
    DATA_API = "https://data-api.polymarket.com"
    
    def __init__(
        self,
        config: BotConfig,
        db_manager=None,
        cache_manager=None
    ):
        super().__init__(config, db_manager, cache_manager)
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_trade_id: str = ""
        self._trades_buffer: List[TradeRecord] = []
        self._trades_by_market: Dict[str, List[TradeRecord]] = defaultdict(list)
        self._whale_threshold = 10000  # $10k
    
    async def start(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        await super().start()
    
    async def stop(self):
        await super().stop()
        if self._session:
            await self._session.close()
    
    async def collect(self) -> List[TradeRecord]:
        """Collect recent trades."""
        try:
            trades = await self._fetch_trades()
            
            new_trades = []
            for trade in trades:
                record = self._parse_trade(trade)
                if record and record.trade_id != self._last_trade_id:
                    new_trades.append(record)
                    self._trades_buffer.append(record)
                    self._trades_by_market[record.market_id].append(record)
            
            if new_trades:
                self._last_trade_id = new_trades[0].trade_id
                
                # Trim buffers
                self._trades_buffer = self._trades_buffer[-10000:]
                for market_id in self._trades_by_market:
                    self._trades_by_market[market_id] = \
                        self._trades_by_market[market_id][-1000:]
                
                # Publish to cache
                if self.cache:
                    await self.cache.publish(
                        "trades",
                        [t.to_dict() for t in new_trades]
                    )
                    
                    # Detect and publish whale trades
                    whale_trades = [t for t in new_trades if t.value >= self._whale_threshold]
                    if whale_trades:
                        await self.cache.publish(
                            "whale_trades",
                            [t.to_dict() for t in whale_trades]
                        )
                
                logger.info(f"Collected {len(new_trades)} new trades")
            
            return new_trades
            
        except Exception as e:
            logger.error(f"Trade collection error: {e}")
            return []
    
    async def _fetch_trades(self, limit: int = 200) -> List[Dict]:
        """Fetch trades from Data API."""
        try:
            url = f"{self.DATA_API}/trades"
            params = {"limit": limit}
            
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []
        except Exception as e:
            logger.error(f"Fetch trades error: {e}")
            return []
    
    async def fetch_market_trades(
        self, 
        market_id: str, 
        limit: int = 100
    ) -> List[Dict]:
        """Fetch trades for specific market."""
        try:
            url = f"{self.DATA_API}/trades"
            params = {"market": market_id, "limit": limit}
            
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []
        except Exception as e:
            logger.error(f"Fetch market trades error: {e}")
            return []
    
    def _parse_trade(self, data: Dict) -> Optional[TradeRecord]:
        """Parse API response to TradeRecord."""
        try:
            return TradeRecord(
                trade_id=data.get("id", str(data.get("timestamp", ""))),
                market_id=data.get("market", data.get("asset_id", "")),
                outcome=data.get("outcome", ""),
                side=data.get("side", "BUY").upper(),
                price=float(data.get("price", 0)),
                size=float(data.get("size", data.get("amount", 0))),
                value=float(data.get("price", 0)) * float(data.get("size", data.get("amount", 0))),
                maker=data.get("maker", ""),
                taker=data.get("taker", ""),
                timestamp=datetime.fromisoformat(
                    data.get("timestamp", datetime.utcnow().isoformat()).replace("Z", "")
                ) if data.get("timestamp") else datetime.utcnow()
            )
        except Exception as e:
            logger.debug(f"Parse trade error: {e}")
            return None
    
    def get_recent_trades(
        self, 
        market_id: str = None, 
        minutes: int = 60
    ) -> List[TradeRecord]:
        """Get recent trades."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        if market_id:
            trades = self._trades_by_market.get(market_id, [])
        else:
            trades = self._trades_buffer
        
        return [t for t in trades if t.timestamp > cutoff]
    
    def calculate_vwap(
        self, 
        market_id: str, 
        minutes: int = 60
    ) -> float:
        """Calculate VWAP for market."""
        trades = self.get_recent_trades(market_id, minutes)
        if not trades:
            return 0.0
        
        total_value = sum(t.price * t.size for t in trades)
        total_size = sum(t.size for t in trades)
        
        return total_value / total_size if total_size > 0 else 0.0
    
    def analyze_flow(
        self, 
        market_id: str = None, 
        minutes: int = 60
    ) -> Dict[str, Any]:
        """Analyze trade flow."""
        trades = self.get_recent_trades(market_id, minutes)
        if not trades:
            return {}
        
        buy_volume = sum(t.value for t in trades if t.side == "BUY")
        sell_volume = sum(t.value for t in trades if t.side == "SELL")
        total_volume = buy_volume + sell_volume
        
        buy_count = len([t for t in trades if t.side == "BUY"])
        sell_count = len([t for t in trades if t.side == "SELL"])
        
        avg_trade_size = sum(t.value for t in trades) / len(trades) if trades else 0
        
        # Whale activity
        whale_trades = [t for t in trades if t.value >= self._whale_threshold]
        whale_volume = sum(t.value for t in whale_trades)
        
        return {
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "total_volume": total_volume,
            "net_flow": buy_volume - sell_volume,
            "flow_ratio": buy_volume / sell_volume if sell_volume > 0 else float('inf'),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "trade_count": len(trades),
            "avg_trade_size": avg_trade_size,
            "whale_trades": len(whale_trades),
            "whale_volume": whale_volume,
            "whale_pct": whale_volume / total_volume * 100 if total_volume > 0 else 0,
            "vwap": self.calculate_vwap(market_id, minutes),
            "pressure": "buy" if buy_volume > sell_volume else "sell"
        }
    
    def detect_whales(
        self, 
        threshold: float = None,
        minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Detect whale traders from recent activity."""
        threshold = threshold or self._whale_threshold
        trades = self.get_recent_trades(minutes=minutes)
        
        # Group by trader
        trader_stats = defaultdict(lambda: {
            "total_volume": 0,
            "trade_count": 0,
            "markets": set(),
            "buy_volume": 0,
            "sell_volume": 0
        })
        
        for trade in trades:
            for addr in [trade.maker, trade.taker]:
                if addr:
                    stats = trader_stats[addr]
                    stats["total_volume"] += trade.value
                    stats["trade_count"] += 1
                    stats["markets"].add(trade.market_id)
                    if trade.side == "BUY":
                        stats["buy_volume"] += trade.value
                    else:
                        stats["sell_volume"] += trade.value
        
        # Filter whales
        whales = []
        for addr, stats in trader_stats.items():
            if stats["total_volume"] >= threshold:
                whales.append({
                    "address": addr,
                    "total_volume": stats["total_volume"],
                    "trade_count": stats["trade_count"],
                    "markets_traded": len(stats["markets"]),
                    "buy_volume": stats["buy_volume"],
                    "sell_volume": stats["sell_volume"],
                    "net_flow": stats["buy_volume"] - stats["sell_volume"]
                })
        
        return sorted(whales, key=lambda x: x["total_volume"], reverse=True)

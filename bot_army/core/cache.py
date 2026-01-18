"""Redis cache and pub/sub management."""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import timedelta

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

logger = logging.getLogger("cache")


class CacheManager:
    """Manages Redis connections for caching and pub/sub."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._subscribers: Dict[str, List[Callable]] = {}
        self._listener_task: Optional[asyncio.Task] = None
    
    async def connect(self):
        """Connect to Redis."""
        if redis is None:
            logger.warning("redis not installed - using mock cache")
            return
        
        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True
        )
        await self._client.ping()
        logger.info(f"Connected to Redis at {self.host}:{self.port}")
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self._listener_task:
            self._listener_task.cancel()
        if self._pubsub:
            await self._pubsub.close()
        if self._client:
            await self._client.close()
        logger.info("Disconnected from Redis")
    
    # ========== Cache Operations ==========
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._client:
            return None
        value = await self._client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ):
        """Set value in cache."""
        if not self._client:
            return
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        if ttl:
            await self._client.setex(key, ttl, value)
        else:
            await self._client.set(key, value)
    
    async def delete(self, key: str):
        """Delete key from cache."""
        if self._client:
            await self._client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._client:
            return False
        return await self._client.exists(key)
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment counter."""
        if not self._client:
            return 0
        return await self._client.incrby(key, amount)
    
    async def expire(self, key: str, seconds: int):
        """Set key expiration."""
        if self._client:
            await self._client.expire(key, seconds)
    
    # ========== List Operations ==========
    
    async def lpush(self, key: str, *values):
        """Push to left of list."""
        if self._client:
            json_values = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in values]
            await self._client.lpush(key, *json_values)
    
    async def rpush(self, key: str, *values):
        """Push to right of list."""
        if self._client:
            json_values = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in values]
            await self._client.rpush(key, *json_values)
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get list range."""
        if not self._client:
            return []
        values = await self._client.lrange(key, start, end)
        result = []
        for v in values:
            try:
                result.append(json.loads(v))
            except (json.JSONDecodeError, TypeError):
                result.append(v)
        return result
    
    async def ltrim(self, key: str, start: int, end: int):
        """Trim list to range."""
        if self._client:
            await self._client.ltrim(key, start, end)
    
    # ========== Hash Operations ==========
    
    async def hget(self, name: str, key: str) -> Optional[Any]:
        """Get hash field."""
        if not self._client:
            return None
        value = await self._client.hget(name, key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None
    
    async def hset(self, name: str, key: str, value: Any):
        """Set hash field."""
        if self._client:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            await self._client.hset(name, key, value)
    
    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all hash fields."""
        if not self._client:
            return {}
        data = await self._client.hgetall(name)
        result = {}
        for k, v in data.items():
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                result[k] = v
        return result
    
    # ========== Pub/Sub Operations ==========
    
    async def publish(self, channel: str, message: Any):
        """Publish message to channel."""
        if not self._client:
            return
        if isinstance(message, (dict, list)):
            message = json.dumps(message)
        await self._client.publish(channel, message)
        logger.debug(f"Published to {channel}")
    
    async def subscribe(self, channel: str, callback: Callable):
        """Subscribe to channel with callback."""
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(callback)
        
        if not self._pubsub and self._client:
            self._pubsub = self._client.pubsub()
            await self._pubsub.subscribe(*self._subscribers.keys())
            self._listener_task = asyncio.create_task(self._listen())
        elif self._pubsub:
            await self._pubsub.subscribe(channel)
    
    async def _listen(self):
        """Listen for pub/sub messages."""
        while True:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )
                if message and message['type'] == 'message':
                    channel = message['channel']
                    data = message['data']
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    
                    for callback in self._subscribers.get(channel, []):
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(data)
                            else:
                                callback(data)
                        except Exception as e:
                            logger.error(f"Subscriber error: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pub/sub listen error: {e}")
                await asyncio.sleep(1)
    
    # ========== Stream Operations (for event sourcing) ==========
    
    async def xadd(self, stream: str, data: Dict[str, Any], maxlen: int = 10000):
        """Add to stream."""
        if self._client:
            str_data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                       for k, v in data.items()}
            await self._client.xadd(stream, str_data, maxlen=maxlen)
    
    async def xread(
        self, 
        streams: Dict[str, str], 
        count: int = 100,
        block: int = None
    ) -> List[Dict]:
        """Read from streams."""
        if not self._client:
            return []
        result = await self._client.xread(streams, count=count, block=block)
        return result
    
    async def xrange(
        self,
        stream: str,
        start: str = "-",
        end: str = "+",
        count: int = 100
    ) -> List[Dict]:
        """Get range from stream."""
        if not self._client:
            return []
        return await self._client.xrange(stream, start, end, count=count)

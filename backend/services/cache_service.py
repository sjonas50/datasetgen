import redis.asyncio as redis
import json
from typing import Any, Optional
from datetime import timedelta

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class CacheService:
    """Redis-based caching service for LLM responses and data"""
    
    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self._redis: Optional[redis.Redis] = None
    
    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection"""
        if self._redis is None:
            self._redis = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            r = await self._get_redis()
            value = await r.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = 3600
    ) -> bool:
        """Set value in cache with optional TTL"""
        try:
            r = await self._get_redis()
            serialized = json.dumps(value)
            
            if ttl:
                await r.setex(key, ttl, serialized)
            else:
                await r.set(key, serialized)
            
            return True
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            r = await self._get_redis()
            await r.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            r = await self._get_redis()
            return await r.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {str(e)}")
            return False
    
    async def set_with_tags(
        self, 
        key: str, 
        value: Any, 
        tags: list[str], 
        ttl: Optional[int] = 3600
    ) -> bool:
        """Set value with tags for bulk invalidation"""
        try:
            r = await self._get_redis()
            
            # Set the main value
            await self.set(key, value, ttl)
            
            # Add key to tag sets
            for tag in tags:
                tag_key = f"tag:{tag}"
                await r.sadd(tag_key, key)
                if ttl:
                    await r.expire(tag_key, ttl + 60)  # Slightly longer TTL
            
            return True
        except Exception as e:
            logger.error(f"Cache set with tags error: {str(e)}")
            return False
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all keys with a specific tag"""
        try:
            r = await self._get_redis()
            tag_key = f"tag:{tag}"
            
            # Get all keys with this tag
            keys = await r.smembers(tag_key)
            
            if keys:
                # Delete all keys
                await r.delete(*keys)
                
            # Delete the tag set
            await r.delete(tag_key)
            
            return len(keys)
        except Exception as e:
            logger.error(f"Cache invalidate by tag error: {str(e)}")
            return 0
    
    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
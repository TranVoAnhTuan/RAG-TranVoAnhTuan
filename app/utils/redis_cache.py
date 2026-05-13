import hashlib
import json
import logging

import redis.asyncio as redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class AppCache:
    """Manages application-level caching via Redis."""

    def __init__(self) -> None:
        try:
            self.redis_client: redis.Redis | None = redis.from_url(settings.REDIS_URL, decode_responses=True)
            logger.info(f"✅ App Cache initialized at {settings.REDIS_URL}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize App Cache: {e}")
            self.redis_client = None

    async def check_cache(self, query: str, topic: str) -> tuple[str | None, dict | None]:
        """Check Redis application cache. Returns (cache_key, cached_result) or (key, None)."""
        if self.redis_client is None:
            return None, None

        normalized_query = query.strip().lower()
        hash_str = hashlib.md5(f"{topic}:{normalized_query}".encode()).hexdigest()
        cache_key = f"app_cache:{hash_str}"

        try:
            cached_res = await self.redis_client.get(cache_key)
            if cached_res:
                logger.info("⚡ Application Cache HIT! Returning instantly.")
                return cache_key, json.loads(cached_res)
        except Exception as e:
            logger.error(f"App Cache read error: {e}")

        return cache_key, None

    async def write_cache(self, cache_key: str, result: dict) -> None:
        """Write a result to the Redis application cache."""
        if self.redis_client is None or cache_key is None:
            return
        try:
            ttl = getattr(settings, "CACHE_TTL_SECONDS", 86400)
            await self.redis_client.setex(cache_key, ttl, json.dumps(result))
        except Exception as e:
            logger.error(f"App Cache write error: {e}")

"""
Queue Backends

Storage backends for queues:
- In-memory (development)
- Redis (production)
- Abstract interface
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import asyncio
import json
import logging

from app.queues.base import (
    Message,
    MessagePriority,
    MessageStatus,
    Queue,
    QueueConfig,
    QueueStats,
    QueueFullError,
)

logger = logging.getLogger(__name__)


class QueueBackend(ABC):
    """Abstract queue backend interface."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to backend."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from backend."""
        pass

    @abstractmethod
    async def push(self, queue_name: str, message: Message) -> str:
        """Push message to queue."""
        pass

    @abstractmethod
    async def pop(self, queue_name: str, count: int = 1) -> List[Message]:
        """Pop messages from queue."""
        pass

    @abstractmethod
    async def ack(self, queue_name: str, message_id: str) -> bool:
        """Acknowledge message."""
        pass

    @abstractmethod
    async def nack(self, queue_name: str, message_id: str) -> bool:
        """Negative acknowledge message."""
        pass

    @abstractmethod
    async def length(self, queue_name: str) -> int:
        """Get queue length."""
        pass

    @abstractmethod
    async def delete_queue(self, queue_name: str) -> bool:
        """Delete a queue."""
        pass


class InMemoryBackend(QueueBackend):
    """
    In-memory queue backend.

    Suitable for development and testing.
    """

    def __init__(self):
        self._queues: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """No-op for in-memory."""
        pass

    async def disconnect(self) -> None:
        """Clear all data."""
        self._queues.clear()

    def _ensure_queue(self, queue_name: str) -> Dict[str, Any]:
        """Ensure queue exists."""
        if queue_name not in self._queues:
            self._queues[queue_name] = {
                "messages": {},
                "pending": [],
                "processing": {},
            }
        return self._queues[queue_name]

    async def push(self, queue_name: str, message: Message) -> str:
        """Push message to queue."""
        async with self._lock:
            queue = self._ensure_queue(queue_name)
            queue["messages"][message.id] = message
            queue["pending"].append(message.id)
            return message.id

    async def pop(self, queue_name: str, count: int = 1) -> List[Message]:
        """Pop messages from queue."""
        async with self._lock:
            queue = self._ensure_queue(queue_name)
            messages = []
            now = datetime.utcnow()

            for _ in range(min(count, len(queue["pending"]))):
                if not queue["pending"]:
                    break

                message_id = queue["pending"].pop(0)
                message = queue["messages"].get(message_id)

                if message and message.is_ready():
                    queue["processing"][message_id] = now
                    message.mark_processing()
                    messages.append(message)
                elif message:
                    # Not ready, put back
                    queue["pending"].append(message_id)

            return messages

    async def ack(self, queue_name: str, message_id: str) -> bool:
        """Acknowledge message."""
        async with self._lock:
            queue = self._ensure_queue(queue_name)

            if message_id in queue["processing"]:
                del queue["processing"][message_id]

            if message_id in queue["messages"]:
                del queue["messages"][message_id]
                return True

            return False

    async def nack(self, queue_name: str, message_id: str) -> bool:
        """Return message to queue."""
        async with self._lock:
            queue = self._ensure_queue(queue_name)

            if message_id in queue["processing"]:
                del queue["processing"][message_id]
                queue["pending"].append(message_id)
                return True

            return False

    async def length(self, queue_name: str) -> int:
        """Get pending message count."""
        queue = self._ensure_queue(queue_name)
        return len(queue["pending"])

    async def delete_queue(self, queue_name: str) -> bool:
        """Delete queue."""
        if queue_name in self._queues:
            del self._queues[queue_name]
            return True
        return False

    async def get_message(self, queue_name: str, message_id: str) -> Optional[Message]:
        """Get message by ID."""
        queue = self._ensure_queue(queue_name)
        return queue["messages"].get(message_id)


class RedisBackend(QueueBackend):
    """
    Redis-based queue backend.

    Production-ready with atomic operations.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "queue:",
        visibility_timeout: int = 30,
    ):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.visibility_timeout = visibility_timeout
        self._redis = None

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            logger.info(f"Connected to Redis: {self.redis_url}")
        except ImportError:
            logger.warning("redis package not installed, using mock")
            self._redis = None
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            self._redis = None

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _queue_key(self, queue_name: str) -> str:
        """Get Redis key for queue."""
        return f"{self.key_prefix}{queue_name}"

    def _processing_key(self, queue_name: str) -> str:
        """Get Redis key for processing set."""
        return f"{self.key_prefix}{queue_name}:processing"

    def _message_key(self, queue_name: str, message_id: str) -> str:
        """Get Redis key for message."""
        return f"{self.key_prefix}{queue_name}:msg:{message_id}"

    async def push(self, queue_name: str, message: Message) -> str:
        """Push message to queue using Redis list."""
        if not self._redis:
            raise RuntimeError("Redis not connected")

        # Store message data
        msg_key = self._message_key(queue_name, message.id)
        await self._redis.set(msg_key, message.serialize())

        # Add to queue (use ZADD for priority queue)
        queue_key = self._queue_key(queue_name)
        score = message.priority.value * 1000000 + datetime.utcnow().timestamp()
        await self._redis.zadd(queue_key, {message.id: score})

        return message.id

    async def pop(self, queue_name: str, count: int = 1) -> List[Message]:
        """Pop messages from queue."""
        if not self._redis:
            raise RuntimeError("Redis not connected")

        queue_key = self._queue_key(queue_name)
        processing_key = self._processing_key(queue_name)
        messages = []

        # Use Lua script for atomic pop
        lua_script = """
        local queue_key = KEYS[1]
        local processing_key = KEYS[2]
        local count = tonumber(ARGV[1])
        local now = tonumber(ARGV[2])
        local timeout = tonumber(ARGV[3])

        local message_ids = redis.call('ZRANGE', queue_key, 0, count - 1)
        local result = {}

        for i, msg_id in ipairs(message_ids) do
            redis.call('ZREM', queue_key, msg_id)
            redis.call('ZADD', processing_key, now + timeout, msg_id)
            table.insert(result, msg_id)
        end

        return result
        """

        # Execute script
        message_ids = await self._redis.eval(
            lua_script,
            2,
            queue_key,
            processing_key,
            count,
            int(datetime.utcnow().timestamp()),
            self.visibility_timeout,
        )

        # Load message data
        for msg_id in message_ids:
            msg_key = self._message_key(queue_name, msg_id.decode() if isinstance(msg_id, bytes) else msg_id)
            data = await self._redis.get(msg_key)
            if data:
                message = Message.deserialize(data)
                message.mark_processing()
                messages.append(message)

        return messages

    async def ack(self, queue_name: str, message_id: str) -> bool:
        """Acknowledge message completion."""
        if not self._redis:
            return False

        processing_key = self._processing_key(queue_name)
        msg_key = self._message_key(queue_name, message_id)

        # Remove from processing and delete message
        await self._redis.zrem(processing_key, message_id)
        await self._redis.delete(msg_key)

        return True

    async def nack(self, queue_name: str, message_id: str) -> bool:
        """Return message to queue."""
        if not self._redis:
            return False

        queue_key = self._queue_key(queue_name)
        processing_key = self._processing_key(queue_name)

        # Move from processing back to queue
        score = await self._redis.zscore(processing_key, message_id)
        if score:
            await self._redis.zrem(processing_key, message_id)
            # Add back with current timestamp
            new_score = datetime.utcnow().timestamp()
            await self._redis.zadd(queue_key, {message_id: new_score})
            return True

        return False

    async def length(self, queue_name: str) -> int:
        """Get queue length."""
        if not self._redis:
            return 0

        queue_key = self._queue_key(queue_name)
        return await self._redis.zcard(queue_key)

    async def delete_queue(self, queue_name: str) -> bool:
        """Delete queue and all messages."""
        if not self._redis:
            return False

        queue_key = self._queue_key(queue_name)
        processing_key = self._processing_key(queue_name)

        # Get all message IDs
        all_ids = await self._redis.zrange(queue_key, 0, -1)
        all_ids += await self._redis.zrange(processing_key, 0, -1)

        # Delete all message keys
        for msg_id in all_ids:
            msg_key = self._message_key(queue_name, msg_id.decode() if isinstance(msg_id, bytes) else msg_id)
            await self._redis.delete(msg_key)

        # Delete queue keys
        await self._redis.delete(queue_key)
        await self._redis.delete(processing_key)

        return True

    async def recover_processing(self, queue_name: str) -> int:
        """Recover timed-out processing messages."""
        if not self._redis:
            return 0

        queue_key = self._queue_key(queue_name)
        processing_key = self._processing_key(queue_name)
        now = int(datetime.utcnow().timestamp())

        # Use Lua script for atomic recovery
        lua_script = """
        local queue_key = KEYS[1]
        local processing_key = KEYS[2]
        local now = tonumber(ARGV[1])

        local expired = redis.call('ZRANGEBYSCORE', processing_key, 0, now)
        local count = 0

        for i, msg_id in ipairs(expired) do
            redis.call('ZREM', processing_key, msg_id)
            redis.call('ZADD', queue_key, now, msg_id)
            count = count + 1
        end

        return count
        """

        recovered = await self._redis.eval(
            lua_script,
            2,
            queue_key,
            processing_key,
            now,
        )

        if recovered > 0:
            logger.info(f"Recovered {recovered} timed-out messages in {queue_name}")

        return recovered

    async def get_stats(self, queue_name: str) -> Dict[str, Any]:
        """Get queue statistics from Redis."""
        if not self._redis:
            return {}

        queue_key = self._queue_key(queue_name)
        processing_key = self._processing_key(queue_name)

        pending = await self._redis.zcard(queue_key)
        processing = await self._redis.zcard(processing_key)

        return {
            "pending": pending,
            "processing": processing,
            "total": pending + processing,
        }


class BackendQueue(Queue):
    """
    Queue implementation using backend interface.

    Provides unified API regardless of backend.
    """

    def __init__(
        self,
        config: QueueConfig,
        backend: QueueBackend,
    ):
        super().__init__(config)
        self.backend = backend
        self._stats = QueueStats(queue_name=config.name)

    async def enqueue(self, message: Message) -> str:
        """Add message to queue."""
        return await self.backend.push(self.name, message)

    async def dequeue(self, count: int = 1) -> List[Message]:
        """Get messages from queue."""
        return await self.backend.pop(self.name, count)

    async def ack(self, message_id: str) -> bool:
        """Acknowledge message."""
        result = await self.backend.ack(self.name, message_id)
        if result:
            self._stats.completed_messages += 1
        return result

    async def nack(self, message_id: str, error: Optional[str] = None) -> bool:
        """Negative acknowledge message."""
        result = await self.backend.nack(self.name, message_id)
        if result:
            self._stats.failed_messages += 1
        return result

    async def size(self) -> int:
        """Get queue size."""
        return await self.backend.length(self.name)

    async def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        self._stats.pending_messages = await self.size()
        return self._stats


# Factory function
def create_queue(
    name: str,
    backend_type: str = "memory",
    **kwargs,
) -> Queue:
    """Create queue with specified backend."""
    config = QueueConfig(name=name, **kwargs)

    if backend_type == "memory":
        from app.queues.base import InMemoryQueue
        return InMemoryQueue(config)
    elif backend_type == "redis":
        backend = RedisBackend(**kwargs)
        return BackendQueue(config, backend)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

"""Memory store for conversation data."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib


logger = structlog.get_logger()


class MemoryType(str, Enum):
    """Types of memory entries."""
    MESSAGE = "message"
    ENTITY = "entity"
    FACT = "fact"
    PREFERENCE = "preference"
    ACTION = "action"
    CONTEXT = "context"


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    type: MemoryType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Temporal info
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    # Importance and relevance
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0

    # Embeddings for semantic search
    embedding: Optional[List[float]] = None

    # Relationships
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    related_ids: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "importance": self.importance,
            "access_count": self.access_count,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "related_ids": self.related_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        entry = cls(
            id=data["id"],
            type=MemoryType(data["type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            session_id=data.get("session_id"),
            agent_id=data.get("agent_id"),
            related_ids=data.get("related_ids", []),
        )

        if data.get("created_at"):
            entry.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("accessed_at"):
            entry.accessed_at = datetime.fromisoformat(data["accessed_at"])
        if data.get("expires_at"):
            entry.expires_at = datetime.fromisoformat(data["expires_at"])

        return entry


class MemoryStore:
    """
    Persistent memory store.

    Supports:
    - In-memory storage
    - Redis persistence
    - TTL-based expiration
    - Semantic search (with embeddings)
    """

    def __init__(
        self,
        redis_client=None,
        default_ttl_seconds: int = 86400,  # 24 hours
        max_entries: int = 10000,
    ):
        self._memories: Dict[str, MemoryEntry] = {}
        self._redis = redis_client
        self._default_ttl = default_ttl_seconds
        self._max_entries = max_entries
        self._lock = asyncio.Lock()

        # Indices
        self._by_session: Dict[str, List[str]] = {}
        self._by_type: Dict[MemoryType, List[str]] = {}

    async def store(
        self,
        entry: MemoryEntry,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Store a memory entry.

        Args:
            entry: Memory entry to store
            ttl_seconds: Optional TTL override

        Returns:
            Entry ID
        """
        async with self._lock:
            # Set expiration
            ttl = ttl_seconds or self._default_ttl
            if ttl > 0:
                entry.expires_at = datetime.utcnow() + timedelta(seconds=ttl)

            # Store entry
            self._memories[entry.id] = entry

            # Update indices
            if entry.session_id:
                if entry.session_id not in self._by_session:
                    self._by_session[entry.session_id] = []
                self._by_session[entry.session_id].append(entry.id)

            if entry.type not in self._by_type:
                self._by_type[entry.type] = []
            self._by_type[entry.type].append(entry.id)

            # Persist to Redis
            await self._persist_to_redis(entry, ttl)

            # Check limits
            await self._enforce_limits()

        logger.debug(
            "memory_stored",
            entry_id=entry.id,
            type=entry.type.value,
        )

        return entry.id

    async def retrieve(
        self,
        entry_id: str,
        touch: bool = True,
    ) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry.

        Args:
            entry_id: Entry ID
            touch: Update access time

        Returns:
            Memory entry or None
        """
        entry = self._memories.get(entry_id)

        if not entry:
            # Try Redis
            entry = await self._load_from_redis(entry_id)
            if entry:
                self._memories[entry_id] = entry

        if entry and entry.is_expired():
            await self.delete(entry_id)
            return None

        if entry and touch:
            entry.touch()

        return entry

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        async with self._lock:
            entry = self._memories.pop(entry_id, None)

            if entry:
                # Remove from indices
                if entry.session_id and entry.session_id in self._by_session:
                    self._by_session[entry.session_id] = [
                        id for id in self._by_session[entry.session_id]
                        if id != entry_id
                    ]

                if entry.type in self._by_type:
                    self._by_type[entry.type] = [
                        id for id in self._by_type[entry.type]
                        if id != entry_id
                    ]

                # Delete from Redis
                if self._redis:
                    await self._redis.delete(f"memory:{entry_id}")

                return True

        return False

    async def query(
        self,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
        limit: int = 100,
        order_by: str = "created_at",  # created_at, accessed_at, importance
        descending: bool = True,
    ) -> List[MemoryEntry]:
        """
        Query memory entries.

        Args:
            session_id: Filter by session
            memory_type: Filter by type
            min_importance: Minimum importance
            limit: Maximum results
            order_by: Sort field
            descending: Sort order

        Returns:
            List of matching entries
        """
        # Get candidate IDs
        if session_id and session_id in self._by_session:
            candidate_ids = set(self._by_session[session_id])
        else:
            candidate_ids = set(self._memories.keys())

        if memory_type and memory_type in self._by_type:
            candidate_ids &= set(self._by_type[memory_type])

        # Filter and collect
        entries = []
        for entry_id in candidate_ids:
            entry = await self.retrieve(entry_id, touch=False)
            if not entry:
                continue

            if entry.importance < min_importance:
                continue

            entries.append(entry)

        # Sort
        if order_by == "created_at":
            entries.sort(key=lambda e: e.created_at, reverse=descending)
        elif order_by == "accessed_at":
            entries.sort(key=lambda e: e.accessed_at, reverse=descending)
        elif order_by == "importance":
            entries.sort(key=lambda e: e.importance, reverse=descending)

        return entries[:limit]

    async def search_semantic(
        self,
        query_embedding: List[float],
        session_id: Optional[str] = None,
        limit: int = 10,
        min_similarity: float = 0.7,
    ) -> List[tuple[MemoryEntry, float]]:
        """
        Search memories by semantic similarity.

        Args:
            query_embedding: Query embedding vector
            session_id: Filter by session
            limit: Maximum results
            min_similarity: Minimum cosine similarity

        Returns:
            List of (entry, similarity) tuples
        """
        results = []

        # Get candidates
        if session_id and session_id in self._by_session:
            candidate_ids = self._by_session[session_id]
        else:
            candidate_ids = list(self._memories.keys())

        for entry_id in candidate_ids:
            entry = await self.retrieve(entry_id, touch=False)
            if not entry or not entry.embedding:
                continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, entry.embedding)

            if similarity >= min_similarity:
                results.append((entry, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Calculate cosine similarity between vectors."""
        import math

        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    async def clear_session(self, session_id: str) -> int:
        """Clear all memories for a session."""
        count = 0

        if session_id in self._by_session:
            entry_ids = list(self._by_session[session_id])
            for entry_id in entry_ids:
                if await self.delete(entry_id):
                    count += 1

        return count

    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        count = 0
        expired_ids = []

        for entry_id, entry in list(self._memories.items()):
            if entry.is_expired():
                expired_ids.append(entry_id)

        for entry_id in expired_ids:
            if await self.delete(entry_id):
                count += 1

        if count > 0:
            logger.info("memory_cleanup", removed=count)

        return count

    async def _persist_to_redis(
        self,
        entry: MemoryEntry,
        ttl_seconds: int,
    ) -> None:
        """Persist entry to Redis."""
        if not self._redis:
            return

        try:
            data = entry.to_dict()
            # Don't store embeddings in Redis (too large)
            data.pop("embedding", None)

            await self._redis.setex(
                f"memory:{entry.id}",
                ttl_seconds,
                json.dumps(data),
            )
        except Exception as e:
            logger.error("redis_persist_error", error=str(e))

    async def _load_from_redis(self, entry_id: str) -> Optional[MemoryEntry]:
        """Load entry from Redis."""
        if not self._redis:
            return None

        try:
            data = await self._redis.get(f"memory:{entry_id}")
            if data:
                return MemoryEntry.from_dict(json.loads(data))
        except Exception as e:
            logger.error("redis_load_error", error=str(e))

        return None

    async def _enforce_limits(self) -> None:
        """Enforce memory limits."""
        if len(self._memories) <= self._max_entries:
            return

        # Remove oldest entries
        entries = list(self._memories.values())
        entries.sort(key=lambda e: e.accessed_at)

        to_remove = len(entries) - self._max_entries
        for entry in entries[:to_remove]:
            await self.delete(entry.id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        type_counts = {}
        for memory_type, ids in self._by_type.items():
            type_counts[memory_type.value] = len(ids)

        return {
            "total_entries": len(self._memories),
            "max_entries": self._max_entries,
            "sessions": len(self._by_session),
            "by_type": type_counts,
            "redis_enabled": self._redis is not None,
        }

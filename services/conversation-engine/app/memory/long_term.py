"""Long-term memory for persistent knowledge."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import hashlib

from app.memory.store import MemoryStore, MemoryEntry, MemoryType


logger = structlog.get_logger()


@dataclass
class UserProfile:
    """User profile information."""
    user_id: str
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    history_summary: Optional[str] = None
    first_contact: datetime = field(default_factory=datetime.utcnow)
    last_contact: datetime = field(default_factory=datetime.utcnow)
    total_interactions: int = 0


class LongTermMemory:
    """
    Long-term memory for persistent knowledge.

    Stores:
    - User profiles across sessions
    - Learned facts and preferences
    - Historical patterns
    - Cross-session context
    """

    def __init__(
        self,
        store: MemoryStore,
        agent_id: Optional[str] = None,
    ):
        self._store = store
        self.agent_id = agent_id
        self._user_profiles: Dict[str, UserProfile] = {}

    async def remember_user(
        self,
        user_id: str,
        session_id: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Remember information about a user.

        Args:
            user_id: User identifier (phone, email, etc.)
            session_id: Current session
            data: Information to remember
        """
        # Get or create profile
        profile = await self.get_user_profile(user_id)
        if not profile:
            profile = UserProfile(user_id=user_id)
            self._user_profiles[user_id] = profile

        # Update profile
        profile.last_contact = datetime.utcnow()
        profile.total_interactions += 1

        if "name" in data:
            profile.name = data["name"]
        if "phone" in data:
            profile.phone = data["phone"]
        if "email" in data:
            profile.email = data["email"]
        if "preferences" in data:
            profile.preferences.update(data["preferences"])

        # Store memories
        for key, value in data.items():
            if key in ("name", "phone", "email", "preferences"):
                continue

            entry = MemoryEntry(
                id=f"user:{user_id}:{key}:{session_id}",
                type=MemoryType.FACT,
                content=f"{key}: {value}",
                metadata={
                    "user_id": user_id,
                    "key": key,
                    "value": value,
                },
                session_id=session_id,
                agent_id=self.agent_id,
                importance=0.6,
            )
            await self._store.store(entry, ttl_seconds=86400 * 30)  # 30 days

        # Persist profile
        await self._persist_user_profile(profile)

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile."""
        if user_id in self._user_profiles:
            return self._user_profiles[user_id]

        # Try to load from store
        entry = await self._store.retrieve(f"profile:{user_id}")
        if entry:
            profile = self._deserialize_profile(entry)
            self._user_profiles[user_id] = profile
            return profile

        return None

    async def get_user_context(
        self,
        user_id: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Get context about a user for conversation.

        Args:
            user_id: User identifier
            limit: Max memories to retrieve

        Returns:
            Context dictionary
        """
        profile = await self.get_user_profile(user_id)

        # Get recent memories about this user
        memories = await self._store.query(
            memory_type=MemoryType.FACT,
            min_importance=0.5,
            limit=limit,
            order_by="accessed_at",
        )

        # Filter to this user
        user_memories = [
            m for m in memories
            if m.metadata.get("user_id") == user_id
        ]

        context = {
            "has_history": profile is not None,
            "first_time": profile is None,
        }

        if profile:
            context.update({
                "user_name": profile.name,
                "user_phone": profile.phone,
                "user_email": profile.email,
                "preferences": profile.preferences,
                "total_interactions": profile.total_interactions,
                "history_summary": profile.history_summary,
            })

        if user_memories:
            context["known_facts"] = [
                {"key": m.metadata.get("key"), "value": m.metadata.get("value")}
                for m in user_memories
            ]

        return context

    async def learn_fact(
        self,
        subject: str,
        predicate: str,
        obj: Any,
        session_id: Optional[str] = None,
        importance: float = 0.5,
    ) -> str:
        """
        Learn a new fact.

        Args:
            subject: Fact subject
            predicate: Relationship/predicate
            obj: Fact object
            session_id: Source session
            importance: Importance score

        Returns:
            Memory entry ID
        """
        fact_id = hashlib.md5(
            f"{subject}:{predicate}:{obj}".encode()
        ).hexdigest()[:12]

        entry = MemoryEntry(
            id=f"fact:{fact_id}",
            type=MemoryType.FACT,
            content=f"{subject} {predicate} {obj}",
            metadata={
                "subject": subject,
                "predicate": predicate,
                "object": obj,
            },
            session_id=session_id,
            agent_id=self.agent_id,
            importance=importance,
        )

        return await self._store.store(entry, ttl_seconds=86400 * 90)  # 90 days

    async def learn_preference(
        self,
        user_id: str,
        preference_type: str,
        preference_value: Any,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Learn a user preference.

        Args:
            user_id: User identifier
            preference_type: Type of preference
            preference_value: Preference value
            session_id: Source session
        """
        entry = MemoryEntry(
            id=f"pref:{user_id}:{preference_type}",
            type=MemoryType.PREFERENCE,
            content=f"User prefers {preference_type}: {preference_value}",
            metadata={
                "user_id": user_id,
                "preference_type": preference_type,
                "preference_value": preference_value,
            },
            session_id=session_id,
            agent_id=self.agent_id,
            importance=0.7,
        )

        await self._store.store(entry, ttl_seconds=86400 * 365)  # 1 year

        # Update profile
        profile = await self.get_user_profile(user_id)
        if profile:
            profile.preferences[preference_type] = preference_value
            await self._persist_user_profile(profile)

    async def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get all preferences for a user."""
        profile = await self.get_user_profile(user_id)
        return profile.preferences if profile else {}

    async def recall_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
    ) -> List[MemoryEntry]:
        """
        Recall memories similar to query.

        Args:
            query_embedding: Query embedding vector
            limit: Max results

        Returns:
            List of similar memories
        """
        results = await self._store.search_semantic(
            query_embedding=query_embedding,
            limit=limit,
            min_similarity=0.7,
        )
        return [entry for entry, _ in results]

    async def summarize_history(
        self,
        user_id: str,
        session_id: str,
        summary: str,
    ) -> None:
        """
        Store a summary of conversation history.

        Args:
            user_id: User identifier
            session_id: Session to summarize
            summary: Conversation summary
        """
        entry = MemoryEntry(
            id=f"summary:{user_id}:{session_id}",
            type=MemoryType.CONTEXT,
            content=summary,
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "type": "conversation_summary",
            },
            session_id=session_id,
            agent_id=self.agent_id,
            importance=0.8,
        )

        await self._store.store(entry, ttl_seconds=86400 * 30)  # 30 days

        # Update profile
        profile = await self.get_user_profile(user_id)
        if profile:
            profile.history_summary = summary
            await self._persist_user_profile(profile)

    async def get_history_summaries(
        self,
        user_id: str,
        limit: int = 5,
    ) -> List[str]:
        """Get recent history summaries for a user."""
        entries = await self._store.query(
            memory_type=MemoryType.CONTEXT,
            limit=limit,
            order_by="created_at",
        )

        summaries = []
        for entry in entries:
            if entry.metadata.get("user_id") == user_id:
                if entry.metadata.get("type") == "conversation_summary":
                    summaries.append(entry.content)

        return summaries

    async def _persist_user_profile(self, profile: UserProfile) -> None:
        """Persist user profile to store."""
        entry = MemoryEntry(
            id=f"profile:{profile.user_id}",
            type=MemoryType.ENTITY,
            content=f"User profile for {profile.name or profile.user_id}",
            metadata={
                "user_id": profile.user_id,
                "name": profile.name,
                "phone": profile.phone,
                "email": profile.email,
                "preferences": profile.preferences,
                "history_summary": profile.history_summary,
                "first_contact": profile.first_contact.isoformat(),
                "last_contact": profile.last_contact.isoformat(),
                "total_interactions": profile.total_interactions,
            },
            agent_id=self.agent_id,
            importance=0.9,
        )

        await self._store.store(entry, ttl_seconds=86400 * 365)  # 1 year

    def _deserialize_profile(self, entry: MemoryEntry) -> UserProfile:
        """Deserialize user profile from memory entry."""
        meta = entry.metadata
        return UserProfile(
            user_id=meta["user_id"],
            name=meta.get("name"),
            phone=meta.get("phone"),
            email=meta.get("email"),
            preferences=meta.get("preferences", {}),
            history_summary=meta.get("history_summary"),
            first_contact=datetime.fromisoformat(meta["first_contact"]) if meta.get("first_contact") else datetime.utcnow(),
            last_contact=datetime.fromisoformat(meta["last_contact"]) if meta.get("last_contact") else datetime.utcnow(),
            total_interactions=meta.get("total_interactions", 0),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get long-term memory statistics."""
        return {
            "cached_profiles": len(self._user_profiles),
            "agent_id": self.agent_id,
            "store_stats": self._store.get_statistics(),
        }

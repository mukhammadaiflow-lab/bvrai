"""Working memory for active processing."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = structlog.get_logger()


class AttentionLevel(str, Enum):
    """Attention levels for working memory items."""
    FOCUS = "focus"  # Currently being processed
    ACTIVE = "active"  # Recently relevant
    BACKGROUND = "background"  # Passively available
    DORMANT = "dormant"  # About to be forgotten


@dataclass
class WorkingMemoryItem:
    """An item in working memory."""
    id: str
    content: Any
    attention: AttentionLevel = AttentionLevel.ACTIVE
    added_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    decay_rate: float = 0.1  # How fast attention decays
    tags: Set[str] = field(default_factory=set)

    def access(self) -> None:
        """Access this item, refreshing attention."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
        self.attention = AttentionLevel.FOCUS

    def decay(self) -> None:
        """Apply attention decay."""
        levels = [
            AttentionLevel.FOCUS,
            AttentionLevel.ACTIVE,
            AttentionLevel.BACKGROUND,
            AttentionLevel.DORMANT,
        ]
        current_idx = levels.index(self.attention)
        if current_idx < len(levels) - 1:
            self.attention = levels[current_idx + 1]


class WorkingMemory:
    """
    Working memory for active conversation processing.

    Simulates human working memory with:
    - Limited capacity
    - Attention-based prioritization
    - Automatic decay
    - Active processing focus
    """

    def __init__(
        self,
        capacity: int = 7,  # Miller's magic number
        decay_interval_seconds: float = 5.0,
    ):
        self.capacity = capacity
        self.decay_interval = decay_interval_seconds

        self._items: Dict[str, WorkingMemoryItem] = {}
        self._focus_stack: List[str] = []  # Stack of focused items
        self._decay_task: Optional[asyncio.Task] = None
        self._running = False

        # Processing state
        self._current_goal: Optional[str] = None
        self._pending_actions: List[Dict[str, Any]] = []
        self._active_functions: Set[str] = set()

    async def start(self) -> None:
        """Start working memory (enables decay)."""
        self._running = True
        self._decay_task = asyncio.create_task(self._decay_loop())

    async def stop(self) -> None:
        """Stop working memory."""
        self._running = False
        if self._decay_task:
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass

    def hold(
        self,
        item_id: str,
        content: Any,
        tags: Optional[Set[str]] = None,
    ) -> bool:
        """
        Hold an item in working memory.

        Args:
            item_id: Unique item identifier
            content: Item content
            tags: Optional tags for categorization

        Returns:
            True if item was added
        """
        # If already exists, refresh
        if item_id in self._items:
            self._items[item_id].content = content
            self._items[item_id].access()
            return True

        # Check capacity
        if len(self._items) >= self.capacity:
            self._evict_one()

        # Add item
        item = WorkingMemoryItem(
            id=item_id,
            content=content,
            attention=AttentionLevel.ACTIVE,
            tags=tags or set(),
        )
        self._items[item_id] = item

        logger.debug(
            "working_memory_hold",
            item_id=item_id,
            capacity_used=f"{len(self._items)}/{self.capacity}",
        )

        return True

    def focus(self, item_id: str) -> bool:
        """
        Focus attention on an item.

        Args:
            item_id: Item to focus on

        Returns:
            True if item exists and was focused
        """
        if item_id not in self._items:
            return False

        item = self._items[item_id]
        item.access()

        # Update focus stack
        if item_id in self._focus_stack:
            self._focus_stack.remove(item_id)
        self._focus_stack.append(item_id)

        # Demote other items from focus
        for other_id, other_item in self._items.items():
            if other_id != item_id and other_item.attention == AttentionLevel.FOCUS:
                other_item.attention = AttentionLevel.ACTIVE

        return True

    def retrieve(self, item_id: str) -> Optional[Any]:
        """
        Retrieve an item from working memory.

        Args:
            item_id: Item to retrieve

        Returns:
            Item content or None
        """
        if item_id not in self._items:
            return None

        item = self._items[item_id]
        item.access()
        return item.content

    def forget(self, item_id: str) -> bool:
        """
        Forget an item from working memory.

        Args:
            item_id: Item to forget

        Returns:
            True if item was removed
        """
        if item_id in self._items:
            del self._items[item_id]
            if item_id in self._focus_stack:
                self._focus_stack.remove(item_id)
            return True
        return False

    def find_by_tag(self, tag: str) -> List[WorkingMemoryItem]:
        """Find items with a specific tag."""
        return [
            item for item in self._items.values()
            if tag in item.tags
        ]

    def get_focused(self) -> Optional[WorkingMemoryItem]:
        """Get currently focused item."""
        if self._focus_stack:
            item_id = self._focus_stack[-1]
            return self._items.get(item_id)
        return None

    def get_active_items(self) -> List[WorkingMemoryItem]:
        """Get all active (non-dormant) items."""
        return [
            item for item in self._items.values()
            if item.attention != AttentionLevel.DORMANT
        ]

    def set_goal(self, goal: str) -> None:
        """Set current processing goal."""
        self._current_goal = goal

    def get_goal(self) -> Optional[str]:
        """Get current goal."""
        return self._current_goal

    def add_pending_action(self, action: Dict[str, Any]) -> None:
        """Add a pending action to execute."""
        self._pending_actions.append(action)

    def pop_pending_action(self) -> Optional[Dict[str, Any]]:
        """Pop next pending action."""
        if self._pending_actions:
            return self._pending_actions.pop(0)
        return None

    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """Get all pending actions."""
        return list(self._pending_actions)

    def start_function(self, function_name: str) -> None:
        """Mark a function as actively executing."""
        self._active_functions.add(function_name)

    def end_function(self, function_name: str) -> None:
        """Mark a function as completed."""
        self._active_functions.discard(function_name)

    def is_function_active(self, function_name: str) -> bool:
        """Check if a function is active."""
        return function_name in self._active_functions

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get snapshot of working memory state.

        Useful for debugging and context building.
        """
        items_summary = []
        for item in self._items.values():
            items_summary.append({
                "id": item.id,
                "attention": item.attention.value,
                "content_type": type(item.content).__name__,
                "access_count": item.access_count,
                "tags": list(item.tags),
            })

        return {
            "capacity": self.capacity,
            "used": len(self._items),
            "items": items_summary,
            "focus_stack": self._focus_stack.copy(),
            "current_goal": self._current_goal,
            "pending_actions": len(self._pending_actions),
            "active_functions": list(self._active_functions),
        }

    def _evict_one(self) -> None:
        """Evict one item to make room."""
        if not self._items:
            return

        # Find lowest attention item with least accesses
        candidates = sorted(
            self._items.values(),
            key=lambda x: (
                [AttentionLevel.DORMANT, AttentionLevel.BACKGROUND,
                 AttentionLevel.ACTIVE, AttentionLevel.FOCUS].index(x.attention),
                x.access_count,
            ),
        )

        if candidates:
            victim = candidates[0]
            self.forget(victim.id)
            logger.debug(
                "working_memory_evict",
                item_id=victim.id,
                reason="capacity",
            )

    async def _decay_loop(self) -> None:
        """Periodic decay of attention levels."""
        while self._running:
            try:
                await asyncio.sleep(self.decay_interval)

                # Apply decay to non-focused items
                for item in self._items.values():
                    if item.attention != AttentionLevel.FOCUS:
                        item.decay()

                # Remove dormant items
                dormant = [
                    item_id for item_id, item in self._items.items()
                    if item.attention == AttentionLevel.DORMANT
                ]
                for item_id in dormant:
                    self.forget(item_id)
                    logger.debug("working_memory_decay", item_id=item_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("working_memory_decay_error", error=str(e))

    def clear(self) -> None:
        """Clear all working memory."""
        self._items.clear()
        self._focus_stack.clear()
        self._current_goal = None
        self._pending_actions.clear()
        self._active_functions.clear()

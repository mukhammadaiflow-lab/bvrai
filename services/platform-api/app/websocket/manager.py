"""WebSocket connection manager."""

from typing import Optional, Dict, Any, List, Set, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging

from app.websocket.handler import (
    WebSocketConnection,
    WebSocketMessage,
    MessageType,
    ConnectionState,
)

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections.

    Tracks all active connections, supports broadcasting, and handles cleanup.

    Usage:
        manager = ConnectionManager()

        # Add connection
        await manager.add(connection)

        # Broadcast to all
        await manager.broadcast(message)

        # Send to specific user
        await manager.send_to_user(user_id, message)
    """

    def __init__(
        self,
        max_connections: int = 10000,
        cleanup_interval: float = 60.0,
        connection_timeout: float = 300.0,
    ):
        self.max_connections = max_connections
        self.cleanup_interval = cleanup_interval
        self.connection_timeout = connection_timeout

        self._connections: Dict[str, WebSocketConnection] = {}
        self._user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self._session_connections: Dict[str, str] = {}  # session_id -> connection_id
        self._call_connections: Dict[str, str] = {}  # call_id -> connection_id
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the connection manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Connection manager started")

    async def stop(self) -> None:
        """Stop the connection manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for connection in list(self._connections.values()):
            try:
                await connection.websocket.close(
                    code=1001, reason="Server shutdown"
                )
            except Exception:
                pass

        self._connections.clear()
        logger.info("Connection manager stopped")

    async def add(self, connection: WebSocketConnection) -> bool:
        """Add a connection."""
        async with self._lock:
            if len(self._connections) >= self.max_connections:
                logger.warning("Max connections reached")
                return False

            self._connections[connection.connection_id] = connection

            # Index by user
            if connection.user_id:
                if connection.user_id not in self._user_connections:
                    self._user_connections[connection.user_id] = set()
                self._user_connections[connection.user_id].add(connection.connection_id)

            # Index by session
            if connection.session_id:
                self._session_connections[connection.session_id] = connection.connection_id

            # Index by call
            if connection.call_id:
                self._call_connections[connection.call_id] = connection.connection_id

            logger.info(
                f"Connection added: {connection.connection_id}, "
                f"total={len(self._connections)}"
            )
            return True

    async def remove(self, connection_id: str) -> Optional[WebSocketConnection]:
        """Remove a connection."""
        async with self._lock:
            connection = self._connections.pop(connection_id, None)
            if not connection:
                return None

            # Remove from user index
            if connection.user_id and connection.user_id in self._user_connections:
                self._user_connections[connection.user_id].discard(connection_id)
                if not self._user_connections[connection.user_id]:
                    del self._user_connections[connection.user_id]

            # Remove from session index
            if connection.session_id:
                self._session_connections.pop(connection.session_id, None)

            # Remove from call index
            if connection.call_id:
                self._call_connections.pop(connection.call_id, None)

            logger.info(
                f"Connection removed: {connection_id}, "
                f"total={len(self._connections)}"
            )
            return connection

    def get(self, connection_id: str) -> Optional[WebSocketConnection]:
        """Get a connection by ID."""
        return self._connections.get(connection_id)

    def get_by_user(self, user_id: str) -> List[WebSocketConnection]:
        """Get all connections for a user."""
        connection_ids = self._user_connections.get(user_id, set())
        return [
            self._connections[cid] for cid in connection_ids
            if cid in self._connections
        ]

    def get_by_session(self, session_id: str) -> Optional[WebSocketConnection]:
        """Get connection for a session."""
        connection_id = self._session_connections.get(session_id)
        if connection_id:
            return self._connections.get(connection_id)
        return None

    def get_by_call(self, call_id: str) -> Optional[WebSocketConnection]:
        """Get connection for a call."""
        connection_id = self._call_connections.get(call_id)
        if connection_id:
            return self._connections.get(connection_id)
        return None

    def get_all(self) -> List[WebSocketConnection]:
        """Get all connections."""
        return list(self._connections.values())

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._connections)

    async def send(
        self,
        connection_id: str,
        message: WebSocketMessage,
    ) -> bool:
        """Send message to a specific connection."""
        connection = self.get(connection_id)
        if not connection:
            return False

        try:
            await connection.websocket.send_text(message.to_json())
            connection.update_activity()
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            await self.remove(connection_id)
            return False

    async def send_to_user(
        self,
        user_id: str,
        message: WebSocketMessage,
    ) -> int:
        """Send message to all connections of a user."""
        connections = self.get_by_user(user_id)
        sent = 0
        for connection in connections:
            if await self.send(connection.connection_id, message):
                sent += 1
        return sent

    async def send_to_session(
        self,
        session_id: str,
        message: WebSocketMessage,
    ) -> bool:
        """Send message to session connection."""
        connection = self.get_by_session(session_id)
        if connection:
            return await self.send(connection.connection_id, message)
        return False

    async def send_to_call(
        self,
        call_id: str,
        message: WebSocketMessage,
    ) -> bool:
        """Send message to call connection."""
        connection = self.get_by_call(call_id)
        if connection:
            return await self.send(connection.connection_id, message)
        return False

    async def broadcast(
        self,
        message: WebSocketMessage,
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """Broadcast message to all connections."""
        exclude = exclude or set()
        sent = 0

        for connection_id, connection in list(self._connections.items()):
            if connection_id in exclude:
                continue
            if await self.send(connection_id, message):
                sent += 1

        return sent

    async def broadcast_to_users(
        self,
        user_ids: List[str],
        message: WebSocketMessage,
    ) -> int:
        """Broadcast message to specific users."""
        sent = 0
        for user_id in user_ids:
            sent += await self.send_to_user(user_id, message)
        return sent

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale connections."""
        while True:
            await asyncio.sleep(self.cleanup_interval)

            try:
                cutoff = datetime.utcnow() - timedelta(seconds=self.connection_timeout)
                stale = []

                for conn_id, conn in self._connections.items():
                    if conn.last_activity < cutoff:
                        stale.append(conn_id)

                for conn_id in stale:
                    connection = await self.remove(conn_id)
                    if connection:
                        try:
                            await connection.websocket.close(
                                code=1000, reason="Connection timeout"
                            )
                        except Exception:
                            pass

                if stale:
                    logger.info(f"Cleaned up {len(stale)} stale connections")

            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        states = {}
        for conn in self._connections.values():
            state = conn.state.value
            states[state] = states.get(state, 0) + 1

        return {
            "total_connections": len(self._connections),
            "unique_users": len(self._user_connections),
            "active_sessions": len(self._session_connections),
            "active_calls": len(self._call_connections),
            "states": states,
        }


class RoomManager:
    """
    Manages WebSocket rooms for group communication.

    Usage:
        rooms = RoomManager(connection_manager)

        # Join room
        await rooms.join("room-123", connection_id)

        # Send to room
        await rooms.send_to_room("room-123", message)
    """

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self._rooms: Dict[str, Set[str]] = {}  # room_id -> connection_ids
        self._connection_rooms: Dict[str, Set[str]] = {}  # connection_id -> room_ids
        self._lock = asyncio.Lock()

    async def create(self, room_id: str) -> None:
        """Create a room."""
        async with self._lock:
            if room_id not in self._rooms:
                self._rooms[room_id] = set()
                logger.info(f"Room created: {room_id}")

    async def delete(self, room_id: str) -> None:
        """Delete a room and remove all members."""
        async with self._lock:
            members = self._rooms.pop(room_id, set())
            for conn_id in members:
                if conn_id in self._connection_rooms:
                    self._connection_rooms[conn_id].discard(room_id)
            logger.info(f"Room deleted: {room_id}")

    async def join(self, room_id: str, connection_id: str) -> None:
        """Add connection to room."""
        async with self._lock:
            if room_id not in self._rooms:
                self._rooms[room_id] = set()

            self._rooms[room_id].add(connection_id)

            if connection_id not in self._connection_rooms:
                self._connection_rooms[connection_id] = set()
            self._connection_rooms[connection_id].add(room_id)

            logger.debug(f"Connection {connection_id} joined room {room_id}")

    async def leave(self, room_id: str, connection_id: str) -> None:
        """Remove connection from room."""
        async with self._lock:
            if room_id in self._rooms:
                self._rooms[room_id].discard(connection_id)

            if connection_id in self._connection_rooms:
                self._connection_rooms[connection_id].discard(room_id)

            logger.debug(f"Connection {connection_id} left room {room_id}")

    async def leave_all(self, connection_id: str) -> None:
        """Remove connection from all rooms."""
        async with self._lock:
            rooms = self._connection_rooms.pop(connection_id, set())
            for room_id in rooms:
                if room_id in self._rooms:
                    self._rooms[room_id].discard(connection_id)

    def get_members(self, room_id: str) -> Set[str]:
        """Get all members of a room."""
        return self._rooms.get(room_id, set()).copy()

    def get_rooms(self, connection_id: str) -> Set[str]:
        """Get all rooms a connection is in."""
        return self._connection_rooms.get(connection_id, set()).copy()

    async def send_to_room(
        self,
        room_id: str,
        message: WebSocketMessage,
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """Send message to all members of a room."""
        members = self.get_members(room_id)
        exclude = exclude or set()
        sent = 0

        for conn_id in members:
            if conn_id in exclude:
                continue
            if await self.connection_manager.send(conn_id, message):
                sent += 1

        return sent

    def room_count(self) -> int:
        """Get number of rooms."""
        return len(self._rooms)


class BroadcastManager:
    """
    Manages topic-based broadcasts.

    Allows connections to subscribe to topics and receive broadcasts.

    Usage:
        broadcasts = BroadcastManager(connection_manager)

        # Subscribe
        await broadcasts.subscribe("events.calls", connection_id)

        # Publish
        await broadcasts.publish("events.calls", message)
    """

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self._subscriptions: Dict[str, Set[str]] = {}  # topic -> connection_ids
        self._connection_topics: Dict[str, Set[str]] = {}  # connection_id -> topics
        self._lock = asyncio.Lock()

    async def subscribe(
        self,
        topic: str,
        connection_id: str,
    ) -> None:
        """Subscribe connection to topic."""
        async with self._lock:
            if topic not in self._subscriptions:
                self._subscriptions[topic] = set()
            self._subscriptions[topic].add(connection_id)

            if connection_id not in self._connection_topics:
                self._connection_topics[connection_id] = set()
            self._connection_topics[connection_id].add(topic)

    async def unsubscribe(
        self,
        topic: str,
        connection_id: str,
    ) -> None:
        """Unsubscribe connection from topic."""
        async with self._lock:
            if topic in self._subscriptions:
                self._subscriptions[topic].discard(connection_id)

            if connection_id in self._connection_topics:
                self._connection_topics[connection_id].discard(topic)

    async def unsubscribe_all(self, connection_id: str) -> None:
        """Unsubscribe connection from all topics."""
        async with self._lock:
            topics = self._connection_topics.pop(connection_id, set())
            for topic in topics:
                if topic in self._subscriptions:
                    self._subscriptions[topic].discard(connection_id)

    async def publish(
        self,
        topic: str,
        message: WebSocketMessage,
    ) -> int:
        """Publish message to all subscribers of a topic."""
        subscribers = self._subscriptions.get(topic, set()).copy()
        sent = 0

        for conn_id in subscribers:
            if await self.connection_manager.send(conn_id, message):
                sent += 1

        return sent

    async def publish_pattern(
        self,
        pattern: str,
        message: WebSocketMessage,
    ) -> int:
        """Publish to topics matching pattern (e.g., "events.*")."""
        import fnmatch
        sent = 0

        for topic in self._subscriptions.keys():
            if fnmatch.fnmatch(topic, pattern):
                sent += await self.publish(topic, message)

        return sent

    def get_subscribers(self, topic: str) -> Set[str]:
        """Get all subscribers of a topic."""
        return self._subscriptions.get(topic, set()).copy()

    def get_topics(self, connection_id: str) -> Set[str]:
        """Get all topics a connection is subscribed to."""
        return self._connection_topics.get(connection_id, set()).copy()


# Global connection manager
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get or create the global connection manager."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


async def setup_connection_manager() -> ConnectionManager:
    """Setup and start the connection manager."""
    manager = get_connection_manager()
    await manager.start()
    return manager


async def shutdown_connection_manager() -> None:
    """Shutdown the connection manager."""
    global _connection_manager
    if _connection_manager:
        await _connection_manager.stop()
        _connection_manager = None

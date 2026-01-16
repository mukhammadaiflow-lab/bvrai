"""
WebSocket Module

This module provides WebSocket server and connection management
for real-time communication.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
)

from .base import (
    Connection,
    ConnectionConfig,
    ConnectionState,
    ConnectionType,
    Message,
    MessageType,
)


logger = logging.getLogger(__name__)


@dataclass
class WebSocketConfig:
    """WebSocket server configuration."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    path: str = "/ws"

    # SSL
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None

    # Connection settings
    connection_config: ConnectionConfig = field(default_factory=ConnectionConfig)

    # Limits
    max_connections: int = 10000
    max_connections_per_ip: int = 100

    # Timeouts
    handshake_timeout: int = 10
    close_timeout: int = 5

    # Heartbeat
    heartbeat_interval: int = 30
    heartbeat_timeout: int = 60

    # Authentication
    auth_handler: Optional[Callable[[str], bool]] = None


@dataclass
class WebSocketConnection:
    """WebSocket connection wrapper."""

    # Core connection info
    connection: Connection = field(default_factory=Connection)
    websocket: Any = None  # The actual websocket object

    # Message handling
    message_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=1000))
    pending_acks: Dict[str, asyncio.Future] = field(default_factory=dict)

    # Heartbeat
    last_ping_at: Optional[datetime] = None
    last_pong_at: Optional[datetime] = None
    ping_latency_ms: float = 0.0

    # Tasks
    _receive_task: Optional[asyncio.Task] = None
    _send_task: Optional[asyncio.Task] = None
    _heartbeat_task: Optional[asyncio.Task] = None

    async def send(self, message: Message) -> None:
        """Send a message."""
        if not self.websocket or not self.is_active():
            return

        try:
            data = json.dumps(message.to_dict())
            await self.websocket.send(data)
            self.connection.messages_sent += 1
            self.connection.bytes_sent += len(data)
            self.connection.last_activity_at = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.connection.errors += 1

    async def send_binary(self, data: bytes) -> None:
        """Send binary data."""
        if not self.websocket or not self.is_active():
            return

        try:
            await self.websocket.send(data)
            self.connection.bytes_sent += len(data)
            self.connection.last_activity_at = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error sending binary: {e}")
            self.connection.errors += 1

    async def receive(self) -> Optional[Message]:
        """Receive a message."""
        if not self.websocket or not self.is_active():
            return None

        try:
            data = await self.websocket.recv()
            self.connection.last_activity_at = datetime.utcnow()

            if isinstance(data, str):
                self.connection.messages_received += 1
                self.connection.bytes_received += len(data)
                return Message.from_dict(json.loads(data))
            else:
                # Binary data
                self.connection.bytes_received += len(data)
                return Message(
                    type=MessageType.AUDIO_DATA,
                    binary_data=data,
                )
        except Exception as e:
            logger.debug(f"Receive error: {e}")
            return None

    async def close(self, code: int = 1000, reason: str = "Normal closure") -> None:
        """Close the connection."""
        self.connection.state = ConnectionState.DISCONNECTING

        # Cancel tasks
        for task in [self._receive_task, self._send_task, self._heartbeat_task]:
            if task:
                task.cancel()

        try:
            if self.websocket:
                await self.websocket.close(code, reason)
        except Exception as e:
            logger.debug(f"Error closing websocket: {e}")

        self.connection.state = ConnectionState.DISCONNECTED
        self.connection.disconnected_at = datetime.utcnow()

    def is_active(self) -> bool:
        """Check if connection is active."""
        return self.connection.is_active()

    async def ping(self) -> float:
        """Send ping and measure latency."""
        self.last_ping_at = datetime.utcnow()

        try:
            ping_msg = Message(type=MessageType.PING)
            await self.send(ping_msg)

            # Wait for pong (handled in message loop)
            await asyncio.sleep(0.1)

            if self.last_pong_at and self.last_pong_at > self.last_ping_at:
                self.ping_latency_ms = (
                    self.last_pong_at - self.last_ping_at
                ).total_seconds() * 1000
        except Exception:
            pass

        return self.ping_latency_ms


class ConnectionManager:
    """
    Manages WebSocket connections.

    Handles connection lifecycle, authentication, and message routing.
    """

    def __init__(self, config: Optional[WebSocketConfig] = None):
        """
        Initialize connection manager.

        Args:
            config: WebSocket configuration
        """
        self.config = config or WebSocketConfig()
        self._connections: Dict[str, WebSocketConnection] = {}
        self._connections_by_session: Dict[str, Set[str]] = {}
        self._connections_by_ip: Dict[str, Set[str]] = {}
        self._message_handlers: Dict[MessageType, List[Callable]] = {}
        self._lock = asyncio.Lock()

    async def add_connection(
        self,
        websocket: Any,
        remote_address: Optional[str] = None,
    ) -> WebSocketConnection:
        """
        Add a new connection.

        Args:
            websocket: WebSocket object
            remote_address: Client IP address

        Returns:
            WebSocket connection
        """
        async with self._lock:
            # Check connection limits
            if len(self._connections) >= self.config.max_connections:
                raise ConnectionError("Maximum connections reached")

            if remote_address:
                ip_connections = self._connections_by_ip.get(remote_address, set())
                if len(ip_connections) >= self.config.max_connections_per_ip:
                    raise ConnectionError("Maximum connections per IP reached")

            # Create connection
            connection = Connection(
                type=ConnectionType.WEBSOCKET,
                state=ConnectionState.CONNECTED,
                remote_address=remote_address,
                connected_at=datetime.utcnow(),
            )

            ws_connection = WebSocketConnection(
                connection=connection,
                websocket=websocket,
            )

            # Track connection
            self._connections[connection.id] = ws_connection

            if remote_address:
                if remote_address not in self._connections_by_ip:
                    self._connections_by_ip[remote_address] = set()
                self._connections_by_ip[remote_address].add(connection.id)

            logger.info(f"Connection added: {connection.id} from {remote_address}")

            return ws_connection

    async def remove_connection(self, connection_id: str) -> None:
        """
        Remove a connection.

        Args:
            connection_id: Connection ID
        """
        async with self._lock:
            ws_conn = self._connections.get(connection_id)
            if not ws_conn:
                return

            connection = ws_conn.connection

            # Remove from IP tracking
            if connection.remote_address:
                ip_set = self._connections_by_ip.get(connection.remote_address)
                if ip_set:
                    ip_set.discard(connection_id)

            # Remove from session tracking
            if connection.session_id:
                session_set = self._connections_by_session.get(connection.session_id)
                if session_set:
                    session_set.discard(connection_id)

            # Close connection
            await ws_conn.close()

            # Remove from connections
            del self._connections[connection_id]

            logger.info(f"Connection removed: {connection_id}")

    async def authenticate(
        self,
        connection_id: str,
        token: str,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> bool:
        """
        Authenticate a connection.

        Args:
            connection_id: Connection ID
            token: Authentication token
            client_id: Client identifier
            user_id: User ID
            organization_id: Organization ID

        Returns:
            True if authenticated
        """
        ws_conn = self._connections.get(connection_id)
        if not ws_conn:
            return False

        # Validate token
        if self.config.auth_handler:
            if not self.config.auth_handler(token):
                ws_conn.connection.state = ConnectionState.ERROR
                return False

        # Update connection
        ws_conn.connection.state = ConnectionState.AUTHENTICATED
        ws_conn.connection.authenticated_at = datetime.utcnow()
        ws_conn.connection.client_id = client_id
        ws_conn.connection.user_id = user_id
        ws_conn.connection.organization_id = organization_id

        logger.info(f"Connection authenticated: {connection_id}")

        return True

    async def associate_session(
        self,
        connection_id: str,
        session_id: str,
    ) -> None:
        """
        Associate a connection with a session.

        Args:
            connection_id: Connection ID
            session_id: Session ID
        """
        ws_conn = self._connections.get(connection_id)
        if not ws_conn:
            return

        ws_conn.connection.session_id = session_id

        async with self._lock:
            if session_id not in self._connections_by_session:
                self._connections_by_session[session_id] = set()
            self._connections_by_session[session_id].add(connection_id)

    async def send_to_connection(
        self,
        connection_id: str,
        message: Message,
    ) -> bool:
        """
        Send message to a specific connection.

        Args:
            connection_id: Connection ID
            message: Message to send

        Returns:
            True if sent
        """
        ws_conn = self._connections.get(connection_id)
        if not ws_conn or not ws_conn.is_active():
            return False

        await ws_conn.send(message)
        return True

    async def send_to_session(
        self,
        session_id: str,
        message: Message,
    ) -> int:
        """
        Send message to all connections in a session.

        Args:
            session_id: Session ID
            message: Message to send

        Returns:
            Number of connections reached
        """
        connection_ids = self._connections_by_session.get(session_id, set())
        sent = 0

        for conn_id in connection_ids:
            if await self.send_to_connection(conn_id, message):
                sent += 1

        return sent

    async def broadcast(
        self,
        message: Message,
        authenticated_only: bool = True,
    ) -> int:
        """
        Broadcast message to all connections.

        Args:
            message: Message to send
            authenticated_only: Only send to authenticated connections

        Returns:
            Number of connections reached
        """
        sent = 0

        for ws_conn in self._connections.values():
            if authenticated_only and not ws_conn.connection.is_authenticated():
                continue

            if ws_conn.is_active():
                await ws_conn.send(message)
                sent += 1

        return sent

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[WebSocketConnection, Message], None],
    ) -> None:
        """Register a message handler."""
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        self._message_handlers[message_type].append(handler)

    async def handle_message(
        self,
        connection_id: str,
        message: Message,
    ) -> None:
        """
        Handle a received message.

        Args:
            connection_id: Source connection ID
            message: Received message
        """
        ws_conn = self._connections.get(connection_id)
        if not ws_conn:
            return

        # Handle built-in message types
        if message.type == MessageType.PING:
            pong = Message(type=MessageType.PONG)
            await ws_conn.send(pong)
            return

        elif message.type == MessageType.PONG:
            ws_conn.last_pong_at = datetime.utcnow()
            return

        # Call registered handlers
        handlers = self._message_handlers.get(message.type, [])
        for handler in handlers:
            try:
                await handler(ws_conn, message)
            except Exception as e:
                logger.error(f"Handler error: {e}")

    def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """Get a connection by ID."""
        return self._connections.get(connection_id)

    def get_connections_by_session(self, session_id: str) -> List[WebSocketConnection]:
        """Get all connections for a session."""
        connection_ids = self._connections_by_session.get(session_id, set())
        return [
            self._connections[cid]
            for cid in connection_ids
            if cid in self._connections
        ]

    def get_active_connections(self) -> List[WebSocketConnection]:
        """Get all active connections."""
        return [
            ws_conn for ws_conn in self._connections.values()
            if ws_conn.is_active()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        total = len(self._connections)
        authenticated = sum(
            1 for c in self._connections.values()
            if c.connection.is_authenticated()
        )
        active = sum(
            1 for c in self._connections.values()
            if c.is_active()
        )

        return {
            "total_connections": total,
            "authenticated_connections": authenticated,
            "active_connections": active,
            "sessions": len(self._connections_by_session),
        }


class WebSocketServer:
    """
    WebSocket server for handling client connections.

    Provides a high-level interface for running a WebSocket server.
    """

    def __init__(
        self,
        config: Optional[WebSocketConfig] = None,
        connection_manager: Optional[ConnectionManager] = None,
    ):
        """
        Initialize WebSocket server.

        Args:
            config: Server configuration
            connection_manager: Connection manager instance
        """
        self.config = config or WebSocketConfig()
        self.manager = connection_manager or ConnectionManager(config)
        self._server = None
        self._running = False

        # Handlers
        self._on_connect: Optional[Callable] = None
        self._on_disconnect: Optional[Callable] = None
        self._on_message: Optional[Callable] = None

    async def start(self) -> None:
        """Start the WebSocket server."""
        try:
            import websockets

            ssl_context = None
            if self.config.ssl_cert and self.config.ssl_key:
                import ssl
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ssl_context.load_cert_chain(
                    self.config.ssl_cert,
                    self.config.ssl_key,
                )

            self._server = await websockets.serve(
                self._handle_connection,
                self.config.host,
                self.config.port,
                ssl=ssl_context,
            )

            self._running = True
            logger.info(
                f"WebSocket server started on "
                f"{'wss' if ssl_context else 'ws'}://{self.config.host}:{self.config.port}{self.config.path}"
            )

        except ImportError:
            raise ImportError("websockets package required. Install with: pip install websockets")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("WebSocket server stopped")

    async def _handle_connection(
        self,
        websocket: Any,
        path: str,
    ) -> None:
        """Handle a new WebSocket connection."""
        # Get client IP
        remote_address = None
        if hasattr(websocket, 'remote_address'):
            remote_address = websocket.remote_address[0]

        ws_conn = None

        try:
            # Add connection
            ws_conn = await self.manager.add_connection(
                websocket,
                remote_address,
            )

            # Call connect handler
            if self._on_connect:
                await self._on_connect(ws_conn)

            # Message loop
            async for data in websocket:
                try:
                    if isinstance(data, str):
                        message = Message.from_dict(json.loads(data))
                    else:
                        message = Message(
                            type=MessageType.AUDIO_DATA,
                            binary_data=data,
                        )

                    message.source_id = ws_conn.connection.id

                    # Handle message
                    await self.manager.handle_message(
                        ws_conn.connection.id,
                        message,
                    )

                    # Call message handler
                    if self._on_message:
                        await self._on_message(ws_conn, message)

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {ws_conn.connection.id}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")

        except Exception as e:
            logger.error(f"Connection error: {e}")

        finally:
            # Cleanup
            if ws_conn:
                if self._on_disconnect:
                    await self._on_disconnect(ws_conn)

                await self.manager.remove_connection(ws_conn.connection.id)

    def on_connect(self, handler: Callable) -> Callable:
        """Decorator for connection handler."""
        self._on_connect = handler
        return handler

    def on_disconnect(self, handler: Callable) -> Callable:
        """Decorator for disconnection handler."""
        self._on_disconnect = handler
        return handler

    def on_message(self, handler: Callable) -> Callable:
        """Decorator for message handler."""
        self._on_message = handler
        return handler

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


__all__ = [
    "WebSocketConfig",
    "WebSocketConnection",
    "ConnectionManager",
    "WebSocketServer",
]

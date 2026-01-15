"""
Real-time Manager Module

This module provides the central orchestration layer for all
real-time communication operations.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from .base import (
    Connection,
    ConnectionState,
    ConnectionType,
    Message,
    MessageType,
    MessagePriority,
    Event,
    EventType,
    EventPriority,
    RealtimeSession,
    SessionState,
)
from .websocket import (
    WebSocketServer,
    WebSocketConnection,
    ConnectionManager,
    WebSocketConfig,
)
from .audio_stream import (
    AudioStream,
    AudioStreamManager,
    AudioBuffer,
    AudioFormat,
    AudioStreamConfig,
)
from .events import (
    EventBus,
    EventSubscription,
    EventHandler,
)
from .protocol import (
    VoiceAgentProtocol,
    ProtocolHandler,
    ProtocolFrame,
    FrameType,
    HandshakeMessage,
    AudioMessage,
    TranscriptMessage,
    AgentStateMessage,
    FunctionCallMessage,
    FunctionResultMessage,
    MessageRouter,
)


logger = logging.getLogger(__name__)


@dataclass
class RealtimeConfig:
    """Configuration for real-time manager."""

    # WebSocket settings
    host: str = "0.0.0.0"
    port: int = 8080
    path: str = "/ws"
    max_connections: int = 10000
    ping_interval: float = 30.0
    ping_timeout: float = 10.0

    # Audio settings
    default_sample_rate: int = 16000
    default_channels: int = 1
    default_encoding: str = "pcm16"
    audio_buffer_size: int = 32000
    max_audio_streams: int = 5000

    # Session settings
    session_timeout_seconds: int = 86400  # 24 hours (was 1 hour - too aggressive)
    max_sessions_per_user: int = 10
    require_auth: bool = True

    # Event settings
    max_event_queue_size: int = 10000
    event_batch_size: int = 100

    # Performance settings
    worker_count: int = 4
    message_batch_timeout_ms: int = 10


@dataclass
class SessionContext:
    """Context for a real-time session."""

    session: RealtimeSession
    connection: Optional[Connection] = None
    audio_stream: Optional[AudioStream] = None
    protocol: Optional[VoiceAgentProtocol] = None

    # Call/agent context
    call_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_state: str = "idle"

    # Handlers
    audio_handler: Optional[Callable[[bytes], Coroutine[Any, Any, Optional[bytes]]]] = None
    transcript_handler: Optional[Callable[[str, bool], Coroutine[Any, Any, None]]] = None
    function_handler: Optional[Callable[[str, Dict], Coroutine[Any, Any, Any]]] = None

    # Statistics
    messages_received: int = 0
    messages_sent: int = 0
    audio_bytes_received: int = 0
    audio_bytes_sent: int = 0


class RealtimeSessionManager:
    """
    Manages real-time sessions.

    Handles session lifecycle, authentication, and resource cleanup.
    """

    def __init__(self, config: RealtimeConfig):
        """Initialize session manager."""
        self.config = config
        self._sessions: Dict[str, SessionContext] = {}
        self._user_sessions: Dict[str, Set[str]] = {}
        self._connection_sessions: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionContext:
        """
        Create a new real-time session.

        Args:
            user_id: User ID
            metadata: Session metadata

        Returns:
            Session context
        """
        async with self._lock:
            # Check user session limit
            if user_id:
                user_session_count = len(self._user_sessions.get(user_id, set()))
                if user_session_count >= self.config.max_sessions_per_user:
                    raise ValueError(
                        f"Maximum sessions ({self.config.max_sessions_per_user}) "
                        f"reached for user {user_id}"
                    )

            # Create session
            session = RealtimeSession(
                id=str(uuid.uuid4()),
                user_id=user_id,
                state=SessionState.CREATED,
                metadata=metadata or {},
            )

            # Create audio stream
            audio_stream = AudioStream(
                id=f"audio-{session.id}",
                format=AudioFormat(
                    sample_rate=self.config.default_sample_rate,
                    channels=self.config.default_channels,
                    encoding=self.config.default_encoding,
                ),
            )

            # Create protocol instance
            protocol = VoiceAgentProtocol()

            # Create context
            context = SessionContext(
                session=session,
                audio_stream=audio_stream,
                protocol=protocol,
            )

            # Register session
            self._sessions[session.id] = context

            if user_id:
                if user_id not in self._user_sessions:
                    self._user_sessions[user_id] = set()
                self._user_sessions[user_id].add(session.id)

            logger.info(f"Session created: {session.id}")

            return context

    async def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def get_session_by_connection(
        self,
        connection_id: str,
    ) -> Optional[SessionContext]:
        """Get session by connection ID."""
        session_id = self._connection_sessions.get(connection_id)
        if session_id:
            return self._sessions.get(session_id)
        return None

    async def bind_connection(
        self,
        session_id: str,
        connection: Connection,
    ) -> bool:
        """
        Bind a connection to a session.

        Args:
            session_id: Session ID
            connection: Connection to bind

        Returns:
            True if bound successfully
        """
        context = self._sessions.get(session_id)
        if not context:
            return False

        async with self._lock:
            context.connection = connection
            context.session.connection_id = connection.id
            context.session.state = SessionState.CONNECTED
            self._connection_sessions[connection.id] = session_id

        logger.debug(f"Connection {connection.id} bound to session {session_id}")

        return True

    async def unbind_connection(self, connection_id: str) -> Optional[str]:
        """
        Unbind a connection from its session.

        Args:
            connection_id: Connection ID

        Returns:
            Session ID if was bound
        """
        session_id = self._connection_sessions.pop(connection_id, None)

        if session_id:
            context = self._sessions.get(session_id)
            if context:
                context.connection = None
                context.session.connection_id = None

        return session_id

    async def end_session(self, session_id: str) -> Optional[SessionContext]:
        """
        End a session.

        Args:
            session_id: Session ID

        Returns:
            Ended session context
        """
        async with self._lock:
            context = self._sessions.pop(session_id, None)

            if context:
                # Update state
                context.session.state = SessionState.ENDED
                context.session.ended_at = datetime.utcnow()

                # Clean up user mapping
                user_id = context.session.user_id
                if user_id and user_id in self._user_sessions:
                    self._user_sessions[user_id].discard(session_id)

                # Clean up connection mapping
                if context.connection:
                    self._connection_sessions.pop(context.connection.id, None)

                logger.info(f"Session ended: {session_id}")

        return context

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        now = datetime.utcnow()
        timeout = timedelta(seconds=self.config.session_timeout_seconds)
        expired = []

        for session_id, context in self._sessions.items():
            if context.session.state in (SessionState.ENDED, SessionState.ERROR):
                expired.append(session_id)
            elif now - context.session.created_at > timeout:
                expired.append(session_id)

        for session_id in expired:
            await self.end_session(session_id)

        return len(expired)

    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self._sessions)

    def get_user_session_count(self, user_id: str) -> int:
        """Get session count for a user."""
        return len(self._user_sessions.get(user_id, set()))


class DefaultProtocolHandler(ProtocolHandler):
    """Default protocol handler implementation."""

    def __init__(
        self,
        manager: "RealtimeManager",
    ):
        """Initialize handler."""
        self.manager = manager

    async def on_handshake(
        self,
        connection: Connection,
        message: HandshakeMessage,
    ) -> bool:
        """Handle handshake."""
        # Authenticate if required
        if self.manager.config.require_auth and not message.auth_token:
            logger.warning(f"Connection {connection.id} missing auth token")
            return False

        # Validate token (in practice, verify JWT or API key)
        if message.auth_token:
            connection.user_id = await self.manager.validate_auth_token(
                message.auth_token
            )
            if not connection.user_id:
                return False

        connection.state = ConnectionState.AUTHENTICATED
        connection.metadata["capabilities"] = message.capabilities
        connection.metadata["client_version"] = message.version

        logger.info(f"Connection {connection.id} authenticated")

        return True

    async def on_audio(
        self,
        connection: Connection,
        message: AudioMessage,
    ) -> Optional[AudioMessage]:
        """Handle audio message."""
        context = await self.manager.session_manager.get_session_by_connection(
            connection.id
        )

        if not context:
            return None

        # Write to audio stream
        if context.audio_stream:
            await context.audio_stream.write_inbound(message.audio_data)
            context.audio_bytes_received += len(message.audio_data)

        # Process through handler if set
        if context.audio_handler:
            response = await context.audio_handler(message.audio_data)
            if response:
                context.audio_bytes_sent += len(response)
                return AudioMessage(
                    audio_data=response,
                    sample_rate=message.sample_rate,
                    encoding=message.encoding,
                )

        return None

    async def on_transcript(
        self,
        connection: Connection,
        message: TranscriptMessage,
    ) -> None:
        """Handle transcript message."""
        context = await self.manager.session_manager.get_session_by_connection(
            connection.id
        )

        if not context:
            return

        context.messages_received += 1

        # Process through handler if set
        if context.transcript_handler:
            await context.transcript_handler(message.text, message.is_final)

        # Emit event
        await self.manager.event_bus.emit(
            EventType.TRANSCRIPT_RECEIVED,
            data={
                "session_id": context.session.id,
                "text": message.text,
                "is_final": message.is_final,
                "confidence": message.confidence,
            },
            session_id=context.session.id,
        )

    async def on_function_call(
        self,
        connection: Connection,
        message: FunctionCallMessage,
    ) -> FunctionResultMessage:
        """Handle function call."""
        context = await self.manager.session_manager.get_session_by_connection(
            connection.id
        )

        if not context or not context.function_handler:
            return FunctionResultMessage(
                call_id=message.call_id,
                success=False,
                error="No function handler configured",
            )

        try:
            result = await asyncio.wait_for(
                context.function_handler(message.function_name, message.arguments),
                timeout=message.timeout_ms / 1000,
            )

            return FunctionResultMessage(
                call_id=message.call_id,
                success=True,
                result=result,
            )

        except asyncio.TimeoutError:
            return FunctionResultMessage(
                call_id=message.call_id,
                success=False,
                error="Function call timed out",
            )
        except Exception as e:
            logger.error(f"Function call error: {e}")
            return FunctionResultMessage(
                call_id=message.call_id,
                success=False,
                error=str(e),
            )

    async def on_state_change(
        self,
        connection: Connection,
        message: AgentStateMessage,
    ) -> None:
        """Handle state change."""
        context = await self.manager.session_manager.get_session_by_connection(
            connection.id
        )

        if context:
            context.agent_state = message.state

            # Emit event
            await self.manager.event_bus.emit(
                EventType.STATE_CHANGE,
                data={
                    "session_id": context.session.id,
                    "state": message.state,
                    "previous_state": message.previous_state,
                    "reason": message.reason,
                },
                session_id=context.session.id,
            )


class RealtimeManager:
    """
    Central manager for all real-time operations.

    Orchestrates:
    - WebSocket connections
    - Audio streaming
    - Event distribution
    - Session management
    - Protocol handling
    """

    def __init__(self, config: Optional[RealtimeConfig] = None):
        """
        Initialize real-time manager.

        Args:
            config: Manager configuration
        """
        self.config = config or RealtimeConfig()

        # Core components
        self.session_manager = RealtimeSessionManager(self.config)
        self.event_bus = EventBus(max_queue_size=self.config.max_event_queue_size)
        self.audio_manager = AudioStreamManager(
            max_streams=self.config.max_audio_streams,
        )
        self.message_router = MessageRouter()

        # WebSocket server (initialized on start)
        self._ws_server: Optional[WebSocketServer] = None
        self._connection_manager: Optional[ConnectionManager] = None

        # Protocol handler
        self._protocol_handler = DefaultProtocolHandler(self)

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Auth validator (can be overridden)
        self._auth_validator: Optional[Callable[[str], Coroutine[Any, Any, Optional[str]]]] = None

        # Event handlers
        self._on_connect_handlers: List[Callable] = []
        self._on_disconnect_handlers: List[Callable] = []
        self._on_message_handlers: List[Callable] = []

        # Statistics
        self._stats = {
            "connections_total": 0,
            "connections_active": 0,
            "sessions_total": 0,
            "sessions_active": 0,
            "messages_received": 0,
            "messages_sent": 0,
            "audio_bytes_received": 0,
            "audio_bytes_sent": 0,
            "errors": 0,
        }

    async def start(self) -> None:
        """Start the real-time manager."""
        if self._running:
            return

        self._running = True

        # Create WebSocket components
        ws_config = WebSocketConfig(
            host=self.config.host,
            port=self.config.port,
            path=self.config.path,
            max_connections=self.config.max_connections,
            ping_interval=self.config.ping_interval,
            ping_timeout=self.config.ping_timeout,
        )

        self._ws_server = WebSocketServer(ws_config)
        self._connection_manager = ConnectionManager()

        # Register WebSocket handlers
        self._ws_server.on_connect(self._handle_connection)
        self._ws_server.on_message(self._handle_message)
        self._ws_server.on_disconnect(self._handle_disconnection)

        # Start components
        await self.event_bus.start()
        await self._ws_server.start()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(
            f"Realtime manager started on "
            f"{self.config.host}:{self.config.port}{self.config.path}"
        )

    async def stop(self) -> None:
        """Stop the real-time manager."""
        if not self._running:
            return

        self._running = False

        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Stop components
        if self._ws_server:
            await self._ws_server.stop()

        await self.event_bus.stop()

        logger.info("Realtime manager stopped")

    async def _handle_connection(
        self,
        websocket: Any,
        connection: Connection,
    ) -> None:
        """Handle new WebSocket connection."""
        self._stats["connections_total"] += 1
        self._stats["connections_active"] += 1

        if self._connection_manager:
            await self._connection_manager.add_connection(connection, websocket)

        # Notify handlers
        for handler in self._on_connect_handlers:
            try:
                await handler(connection)
            except Exception as e:
                logger.error(f"Connection handler error: {e}")

        # Emit event
        await self.event_bus.emit(
            EventType.CONNECTION_ESTABLISHED,
            data={"connection_id": connection.id},
        )

        logger.debug(f"Connection established: {connection.id}")

    async def _handle_message(
        self,
        connection: Connection,
        message: Union[str, bytes],
    ) -> None:
        """Handle incoming message."""
        self._stats["messages_received"] += 1

        try:
            # Get session context
            context = await self.session_manager.get_session_by_connection(
                connection.id
            )

            if context and context.protocol:
                # Decode through protocol
                if isinstance(message, bytes):
                    frame_type, decoded = await context.protocol.decode_message(
                        message,
                        connection,
                    )
                else:
                    # Text message (JSON)
                    import json
                    data = json.loads(message)
                    frame_type = FrameType.TEXT
                    decoded = data

                    # Route based on message type
                    await self._route_message(connection, context, data)

            # Notify handlers
            for handler in self._on_message_handlers:
                try:
                    await handler(connection, message)
                except Exception as e:
                    logger.error(f"Message handler error: {e}")

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self._stats["errors"] += 1

    async def _route_message(
        self,
        connection: Connection,
        context: SessionContext,
        data: Dict[str, Any],
    ) -> None:
        """Route a JSON message to appropriate handler."""
        msg_type = data.get("type", "")

        if msg_type == "audio":
            # Handle audio data
            audio_data = data.get("audio", b"")
            if isinstance(audio_data, str):
                import base64
                audio_data = base64.b64decode(audio_data)

            if context.audio_stream:
                await context.audio_stream.write_inbound(audio_data)

            if context.audio_handler:
                response = await context.audio_handler(audio_data)
                if response:
                    await self.send_audio(context.session.id, response)

        elif msg_type == "transcript":
            # Handle transcript
            text = data.get("text", "")
            is_final = data.get("is_final", False)

            if context.transcript_handler:
                await context.transcript_handler(text, is_final)

        elif msg_type == "function_call":
            # Handle function call
            call_id = data.get("call_id", "")
            func_name = data.get("function", "")
            args = data.get("arguments", {})

            if context.function_handler:
                result = await context.function_handler(func_name, args)
                await self.send_function_result(context.session.id, call_id, result)

        elif msg_type == "state":
            # Handle state update
            context.agent_state = data.get("state", context.agent_state)

    async def _handle_disconnection(self, connection: Connection) -> None:
        """Handle WebSocket disconnection."""
        self._stats["connections_active"] -= 1

        # Unbind from session
        session_id = await self.session_manager.unbind_connection(connection.id)

        if self._connection_manager:
            await self._connection_manager.remove_connection(connection.id)

        # Notify handlers
        for handler in self._on_disconnect_handlers:
            try:
                await handler(connection)
            except Exception as e:
                logger.error(f"Disconnect handler error: {e}")

        # Emit event
        await self.event_bus.emit(
            EventType.CONNECTION_CLOSED,
            data={
                "connection_id": connection.id,
                "session_id": session_id,
            },
        )

        logger.debug(f"Connection closed: {connection.id}")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes (was every minute)

                # Clean up expired sessions (data is archived, not deleted)
                cleaned = await self.session_manager.cleanup_expired_sessions()
                if cleaned > 0:
                    logger.info(f"Archived {cleaned} expired sessions (data preserved)")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    # Session management

    async def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new real-time session.

        Args:
            user_id: User ID
            metadata: Session metadata

        Returns:
            Session ID
        """
        context = await self.session_manager.create_session(user_id, metadata)

        self._stats["sessions_total"] += 1
        self._stats["sessions_active"] += 1

        # Emit event
        await self.event_bus.emit(
            EventType.SESSION_STARTED,
            data={
                "session_id": context.session.id,
                "user_id": user_id,
            },
            session_id=context.session.id,
        )

        return context.session.id

    async def end_session(self, session_id: str) -> bool:
        """
        End a real-time session.

        Args:
            session_id: Session ID

        Returns:
            True if ended
        """
        context = await self.session_manager.end_session(session_id)

        if context:
            self._stats["sessions_active"] -= 1

            # Close connection if exists
            if context.connection and self._connection_manager:
                await self._connection_manager.close_connection(context.connection.id)

            # Emit event
            await self.event_bus.emit(
                EventType.SESSION_ENDED,
                data={"session_id": session_id},
                session_id=session_id,
            )

            return True

        return False

    async def bind_session_to_connection(
        self,
        session_id: str,
        connection_id: str,
    ) -> bool:
        """
        Bind a session to a connection.

        Args:
            session_id: Session ID
            connection_id: Connection ID

        Returns:
            True if bound
        """
        if not self._connection_manager:
            return False

        connection = await self._connection_manager.get_connection(connection_id)
        if not connection:
            return False

        return await self.session_manager.bind_connection(session_id, connection)

    # Message sending

    async def send_to_session(
        self,
        session_id: str,
        message: Union[str, bytes, Dict[str, Any]],
    ) -> bool:
        """
        Send a message to a session.

        Args:
            session_id: Session ID
            message: Message to send

        Returns:
            True if sent
        """
        context = await self.session_manager.get_session(session_id)

        if not context or not context.connection:
            return False

        if not self._connection_manager:
            return False

        # Convert dict to JSON string
        if isinstance(message, dict):
            import json
            message = json.dumps(message)

        success = await self._connection_manager.send_to_connection(
            context.connection.id,
            message,
        )

        if success:
            self._stats["messages_sent"] += 1
            context.messages_sent += 1

        return success

    async def send_audio(
        self,
        session_id: str,
        audio_data: bytes,
        sample_rate: int = 16000,
        encoding: str = "pcm16",
    ) -> bool:
        """
        Send audio to a session.

        Args:
            session_id: Session ID
            audio_data: Audio bytes
            sample_rate: Sample rate
            encoding: Audio encoding

        Returns:
            True if sent
        """
        context = await self.session_manager.get_session(session_id)

        if not context:
            return False

        # Write to outbound stream
        if context.audio_stream:
            await context.audio_stream.write_outbound(audio_data)

        # Send through WebSocket
        import base64
        message = {
            "type": "audio",
            "audio": base64.b64encode(audio_data).decode(),
            "sample_rate": sample_rate,
            "encoding": encoding,
        }

        success = await self.send_to_session(session_id, message)

        if success:
            self._stats["audio_bytes_sent"] += len(audio_data)
            context.audio_bytes_sent += len(audio_data)

        return success

    async def send_transcript(
        self,
        session_id: str,
        text: str,
        is_final: bool = False,
        speaker: str = "agent",
    ) -> bool:
        """
        Send transcript to a session.

        Args:
            session_id: Session ID
            text: Transcript text
            is_final: Is final transcript
            speaker: Speaker identifier

        Returns:
            True if sent
        """
        message = {
            "type": "transcript",
            "text": text,
            "is_final": is_final,
            "speaker": speaker,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self.send_to_session(session_id, message)

    async def send_function_result(
        self,
        session_id: str,
        call_id: str,
        result: Any,
        success: bool = True,
        error: Optional[str] = None,
    ) -> bool:
        """
        Send function result to a session.

        Args:
            session_id: Session ID
            call_id: Function call ID
            result: Function result
            success: Whether call succeeded
            error: Error message if failed

        Returns:
            True if sent
        """
        message = {
            "type": "function_result",
            "call_id": call_id,
            "success": success,
            "result": result,
            "error": error,
        }

        return await self.send_to_session(session_id, message)

    async def send_state_update(
        self,
        session_id: str,
        state: str,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send state update to a session.

        Args:
            session_id: Session ID
            state: New state
            reason: State change reason
            metadata: Additional metadata

        Returns:
            True if sent
        """
        context = await self.session_manager.get_session(session_id)
        previous_state = context.agent_state if context else ""

        message = {
            "type": "state",
            "state": state,
            "previous_state": previous_state,
            "reason": reason,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        success = await self.send_to_session(session_id, message)

        if success and context:
            context.agent_state = state

        return success

    # Session configuration

    async def set_audio_handler(
        self,
        session_id: str,
        handler: Callable[[bytes], Coroutine[Any, Any, Optional[bytes]]],
    ) -> bool:
        """
        Set audio handler for a session.

        Args:
            session_id: Session ID
            handler: Audio handler function

        Returns:
            True if set
        """
        context = await self.session_manager.get_session(session_id)
        if context:
            context.audio_handler = handler
            return True
        return False

    async def set_transcript_handler(
        self,
        session_id: str,
        handler: Callable[[str, bool], Coroutine[Any, Any, None]],
    ) -> bool:
        """
        Set transcript handler for a session.

        Args:
            session_id: Session ID
            handler: Transcript handler function

        Returns:
            True if set
        """
        context = await self.session_manager.get_session(session_id)
        if context:
            context.transcript_handler = handler
            return True
        return False

    async def set_function_handler(
        self,
        session_id: str,
        handler: Callable[[str, Dict], Coroutine[Any, Any, Any]],
    ) -> bool:
        """
        Set function handler for a session.

        Args:
            session_id: Session ID
            handler: Function handler

        Returns:
            True if set
        """
        context = await self.session_manager.get_session(session_id)
        if context:
            context.function_handler = handler
            return True
        return False

    async def set_call_context(
        self,
        session_id: str,
        call_id: str,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        Set call context for a session.

        Args:
            session_id: Session ID
            call_id: Call ID
            agent_id: Agent ID

        Returns:
            True if set
        """
        context = await self.session_manager.get_session(session_id)
        if context:
            context.call_id = call_id
            context.agent_id = agent_id
            return True
        return False

    # Authentication

    def set_auth_validator(
        self,
        validator: Callable[[str], Coroutine[Any, Any, Optional[str]]],
    ) -> None:
        """
        Set custom authentication validator.

        Args:
            validator: Async function that takes token and returns user_id or None
        """
        self._auth_validator = validator

    async def validate_auth_token(self, token: str) -> Optional[str]:
        """
        Validate authentication token.

        Args:
            token: Auth token

        Returns:
            User ID if valid, None otherwise
        """
        if self._auth_validator:
            return await self._auth_validator(token)

        # Default: accept all tokens and use them as user IDs
        return token if token else None

    # Event subscriptions

    def on_connect(
        self,
        handler: Callable[[Connection], Coroutine[Any, Any, None]],
    ) -> None:
        """Register connection handler."""
        self._on_connect_handlers.append(handler)

    def on_disconnect(
        self,
        handler: Callable[[Connection], Coroutine[Any, Any, None]],
    ) -> None:
        """Register disconnection handler."""
        self._on_disconnect_handlers.append(handler)

    def on_message(
        self,
        handler: Callable[[Connection, Union[str, bytes]], Coroutine[Any, Any, None]],
    ) -> None:
        """Register message handler."""
        self._on_message_handlers.append(handler)

    def subscribe_event(
        self,
        event_type: EventType,
        handler: Callable[[Event], Any],
        **kwargs,
    ) -> str:
        """
        Subscribe to events.

        Args:
            event_type: Event type to subscribe to
            handler: Event handler
            **kwargs: Additional subscription options

        Returns:
            Subscription ID
        """
        return self.event_bus.subscribe(event_type, handler, **kwargs)

    def unsubscribe_event(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        return self.event_bus.unsubscribe(subscription_id)

    # Statistics

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self._stats,
            "sessions_active": self.session_manager.get_active_session_count(),
            "event_bus": self.event_bus.get_stats(),
        }


def create_realtime_manager(
    host: str = "0.0.0.0",
    port: int = 8080,
    **kwargs,
) -> RealtimeManager:
    """
    Create a real-time manager with common configuration.

    Args:
        host: Server host
        port: Server port
        **kwargs: Additional configuration

    Returns:
        Configured real-time manager
    """
    config = RealtimeConfig(
        host=host,
        port=port,
        **kwargs,
    )

    return RealtimeManager(config)


__all__ = [
    "RealtimeConfig",
    "SessionContext",
    "RealtimeSessionManager",
    "DefaultProtocolHandler",
    "RealtimeManager",
    "create_realtime_manager",
]

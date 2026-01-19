"""
SIP Protocol Handler

SIP (Session Initiation Protocol) support:
- SIP client for making/receiving calls
- SIP registrar for registration
- Dialog and transaction management
- Message parsing and generation
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import uuid
import re
import hashlib
import socket

logger = logging.getLogger(__name__)


class SIPMethod(str, Enum):
    """SIP request methods."""
    INVITE = "INVITE"
    ACK = "ACK"
    BYE = "BYE"
    CANCEL = "CANCEL"
    REGISTER = "REGISTER"
    OPTIONS = "OPTIONS"
    PRACK = "PRACK"
    UPDATE = "UPDATE"
    INFO = "INFO"
    SUBSCRIBE = "SUBSCRIBE"
    NOTIFY = "NOTIFY"
    REFER = "REFER"
    MESSAGE = "MESSAGE"


class SIPStatus(int, Enum):
    """SIP response status codes."""
    # 1xx Provisional
    TRYING = 100
    RINGING = 180
    SESSION_PROGRESS = 183

    # 2xx Success
    OK = 200
    ACCEPTED = 202

    # 3xx Redirection
    MOVED_PERMANENTLY = 301
    MOVED_TEMPORARILY = 302

    # 4xx Client Error
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    REQUEST_TIMEOUT = 408
    TEMPORARILY_UNAVAILABLE = 480
    BUSY_HERE = 486
    REQUEST_TERMINATED = 487

    # 5xx Server Error
    SERVER_INTERNAL_ERROR = 500
    NOT_IMPLEMENTED = 501
    SERVICE_UNAVAILABLE = 503

    # 6xx Global Failure
    BUSY_EVERYWHERE = 600
    DECLINE = 603


class DialogState(str, Enum):
    """SIP dialog states."""
    EARLY = "early"
    CONFIRMED = "confirmed"
    TERMINATED = "terminated"


class TransactionState(str, Enum):
    """SIP transaction states."""
    TRYING = "trying"
    PROCEEDING = "proceeding"
    COMPLETED = "completed"
    CONFIRMED = "confirmed"
    TERMINATED = "terminated"


@dataclass
class SIPConfig:
    """SIP configuration."""
    # Local settings
    local_ip: str = "0.0.0.0"
    local_port: int = 5060
    transport: str = "UDP"  # UDP, TCP, TLS

    # User agent
    user_agent: str = "BuilderEngine/1.0"

    # Registration
    register_expires: int = 3600
    min_expires: int = 60

    # Timeouts
    transaction_timeout: float = 32.0
    invite_timeout: float = 180.0
    non_invite_timeout: float = 32.0

    # Retransmissions
    t1: float = 0.5  # RTT estimate
    t2: float = 4.0  # Max retransmit interval
    t4: float = 5.0  # Max duration for message persistence

    # Security
    enable_digest_auth: bool = True
    realm: str = "builderengine.com"

    # Limits
    max_forwards: int = 70
    max_concurrent_calls: int = 100


@dataclass
class SIPURI:
    """SIP URI representation."""
    scheme: str = "sip"
    user: str = ""
    host: str = ""
    port: int = 5060
    parameters: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        """Convert to string."""
        uri = f"{self.scheme}:"
        if self.user:
            uri += f"{self.user}@"
        uri += self.host
        if self.port != 5060:
            uri += f":{self.port}"
        if self.parameters:
            for key, value in self.parameters.items():
                uri += f";{key}={value}"
        return uri

    @classmethod
    def parse(cls, uri_str: str) -> "SIPURI":
        """Parse SIP URI string."""
        uri = cls()

        # Remove angle brackets if present
        uri_str = uri_str.strip("<>")

        # Parse scheme
        if uri_str.startswith("sips:"):
            uri.scheme = "sips"
            uri_str = uri_str[5:]
        elif uri_str.startswith("sip:"):
            uri.scheme = "sip"
            uri_str = uri_str[4:]

        # Parse parameters
        if ";" in uri_str:
            parts = uri_str.split(";")
            uri_str = parts[0]
            for param in parts[1:]:
                if "=" in param:
                    key, value = param.split("=", 1)
                    uri.parameters[key] = value

        # Parse user and host
        if "@" in uri_str:
            uri.user, uri_str = uri_str.split("@", 1)

        # Parse host and port
        if ":" in uri_str:
            uri.host, port_str = uri_str.split(":", 1)
            uri.port = int(port_str)
        else:
            uri.host = uri_str

        return uri


@dataclass
class SIPHeader:
    """SIP header."""
    name: str
    value: str
    parameters: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        """Convert to string."""
        result = f"{self.name}: {self.value}"
        for key, val in self.parameters.items():
            result += f";{key}={val}"
        return result


@dataclass
class SIPMessage:
    """SIP message representation."""
    # Request line (for requests)
    method: Optional[SIPMethod] = None
    request_uri: Optional[SIPURI] = None
    version: str = "SIP/2.0"

    # Status line (for responses)
    status_code: Optional[int] = None
    reason_phrase: Optional[str] = None

    # Headers
    headers: Dict[str, List[str]] = field(default_factory=dict)

    # Body
    body: Optional[str] = None
    content_type: Optional[str] = None

    @property
    def is_request(self) -> bool:
        """Check if message is a request."""
        return self.method is not None

    @property
    def is_response(self) -> bool:
        """Check if message is a response."""
        return self.status_code is not None

    def get_header(self, name: str) -> Optional[str]:
        """Get first header value."""
        values = self.headers.get(name, self.headers.get(name.lower(), []))
        return values[0] if values else None

    def get_headers(self, name: str) -> List[str]:
        """Get all header values."""
        return self.headers.get(name, self.headers.get(name.lower(), []))

    def set_header(self, name: str, value: str) -> None:
        """Set header value."""
        self.headers[name] = [value]

    def add_header(self, name: str, value: str) -> None:
        """Add header value."""
        if name not in self.headers:
            self.headers[name] = []
        self.headers[name].append(value)

    @property
    def call_id(self) -> Optional[str]:
        """Get Call-ID header."""
        return self.get_header("Call-ID")

    @property
    def from_header(self) -> Optional[str]:
        """Get From header."""
        return self.get_header("From")

    @property
    def to_header(self) -> Optional[str]:
        """Get To header."""
        return self.get_header("To")

    @property
    def via_header(self) -> Optional[str]:
        """Get Via header."""
        return self.get_header("Via")

    @property
    def cseq(self) -> Optional[Tuple[int, str]]:
        """Get CSeq header."""
        cseq = self.get_header("CSeq")
        if cseq:
            parts = cseq.split()
            return (int(parts[0]), parts[1])
        return None

    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        lines = []

        # Start line
        if self.is_request:
            lines.append(f"{self.method.value} {self.request_uri} {self.version}")
        else:
            lines.append(f"{self.version} {self.status_code} {self.reason_phrase}")

        # Headers
        for name, values in self.headers.items():
            for value in values:
                lines.append(f"{name}: {value}")

        # Content headers
        if self.body:
            if self.content_type:
                lines.append(f"Content-Type: {self.content_type}")
            lines.append(f"Content-Length: {len(self.body)}")
        else:
            lines.append("Content-Length: 0")

        # Empty line before body
        lines.append("")

        # Body
        if self.body:
            lines.append(self.body)

        return "\r\n".join(lines).encode()

    @classmethod
    def parse(cls, data: bytes) -> "SIPMessage":
        """Parse SIP message from bytes."""
        message = cls()
        text = data.decode("utf-8", errors="ignore")
        lines = text.split("\r\n")

        if not lines:
            return message

        # Parse start line
        start_line = lines[0]
        if start_line.startswith("SIP/"):
            # Response
            parts = start_line.split(" ", 2)
            message.version = parts[0]
            message.status_code = int(parts[1])
            message.reason_phrase = parts[2] if len(parts) > 2 else ""
        else:
            # Request
            parts = start_line.split(" ")
            message.method = SIPMethod(parts[0])
            message.request_uri = SIPURI.parse(parts[1])
            message.version = parts[2] if len(parts) > 2 else "SIP/2.0"

        # Parse headers
        i = 1
        while i < len(lines) and lines[i]:
            line = lines[i]
            if ":" in line:
                name, value = line.split(":", 1)
                message.add_header(name.strip(), value.strip())
            i += 1

        # Parse body
        if i < len(lines) - 1:
            message.body = "\r\n".join(lines[i + 1:])
            message.content_type = message.get_header("Content-Type")

        return message


@dataclass
class SIPDialog:
    """SIP dialog."""
    dialog_id: str
    call_id: str
    local_tag: str
    remote_tag: str

    local_uri: SIPURI = field(default_factory=SIPURI)
    remote_uri: SIPURI = field(default_factory=SIPURI)
    remote_target: SIPURI = field(default_factory=SIPURI)

    local_cseq: int = 0
    remote_cseq: int = 0

    state: DialogState = DialogState.EARLY
    route_set: List[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_next_cseq(self) -> int:
        """Get next local CSeq."""
        self.local_cseq += 1
        return self.local_cseq


@dataclass
class SIPTransaction:
    """SIP transaction."""
    transaction_id: str
    branch: str
    method: SIPMethod

    state: TransactionState = TransactionState.TRYING
    request: Optional[SIPMessage] = None
    responses: List[SIPMessage] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.utcnow)
    retransmit_count: int = 0
    last_retransmit: Optional[datetime] = None


@dataclass
class SIPSession:
    """SIP session."""
    session_id: str
    dialog: SIPDialog
    local_sdp: Optional[str] = None
    remote_sdp: Optional[str] = None

    # State
    state: str = "initial"
    direction: str = "outbound"  # inbound or outbound

    # Endpoints
    caller_uri: SIPURI = field(default_factory=SIPURI)
    callee_uri: SIPURI = field(default_factory=SIPURI)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class SIPMessageBuilder:
    """SIP message builder."""

    def __init__(self, config: SIPConfig):
        self.config = config
        self._cseq = 0

    def _get_cseq(self) -> int:
        """Get next CSeq number."""
        self._cseq += 1
        return self._cseq

    def _generate_branch(self) -> str:
        """Generate Via branch parameter."""
        return f"z9hG4bK{uuid.uuid4().hex[:16]}"

    def _generate_tag(self) -> str:
        """Generate tag parameter."""
        return uuid.uuid4().hex[:8]

    def _generate_call_id(self) -> str:
        """Generate Call-ID."""
        return f"{uuid.uuid4().hex}@{self.config.local_ip}"

    def build_request(
        self,
        method: SIPMethod,
        to_uri: SIPURI,
        from_uri: SIPURI,
        call_id: Optional[str] = None,
        cseq: Optional[int] = None,
        body: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> SIPMessage:
        """Build SIP request."""
        msg = SIPMessage(
            method=method,
            request_uri=to_uri,
            body=body,
            content_type=content_type,
        )

        # Via header
        branch = self._generate_branch()
        via = f"SIP/2.0/{self.config.transport} {self.config.local_ip}:{self.config.local_port};branch={branch}"
        msg.set_header("Via", via)

        # Max-Forwards
        msg.set_header("Max-Forwards", str(self.config.max_forwards))

        # From header
        from_tag = self._generate_tag()
        msg.set_header("From", f"<{from_uri}>;tag={from_tag}")

        # To header
        msg.set_header("To", f"<{to_uri}>")

        # Call-ID
        msg.set_header("Call-ID", call_id or self._generate_call_id())

        # CSeq
        msg.set_header("CSeq", f"{cseq or self._get_cseq()} {method.value}")

        # User-Agent
        msg.set_header("User-Agent", self.config.user_agent)

        return msg

    def build_response(
        self,
        request: SIPMessage,
        status: SIPStatus,
        body: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> SIPMessage:
        """Build SIP response."""
        msg = SIPMessage(
            status_code=status.value,
            reason_phrase=status.name.replace("_", " ").title(),
            body=body,
            content_type=content_type,
        )

        # Copy headers from request
        for header in ["Via", "From", "To", "Call-ID", "CSeq"]:
            values = request.get_headers(header)
            for value in values:
                msg.add_header(header, value)

        # Add To tag if not present
        to_header = msg.get_header("To")
        if to_header and ";tag=" not in to_header:
            tag = self._generate_tag()
            msg.headers["To"] = [f"{to_header};tag={tag}"]

        # User-Agent
        msg.set_header("User-Agent", self.config.user_agent)

        return msg

    def build_invite(
        self,
        to_uri: SIPURI,
        from_uri: SIPURI,
        sdp: str,
    ) -> SIPMessage:
        """Build INVITE request."""
        return self.build_request(
            method=SIPMethod.INVITE,
            to_uri=to_uri,
            from_uri=from_uri,
            body=sdp,
            content_type="application/sdp",
        )

    def build_ack(
        self,
        invite: SIPMessage,
        response: SIPMessage,
    ) -> SIPMessage:
        """Build ACK request."""
        to_uri = invite.request_uri

        msg = self.build_request(
            method=SIPMethod.ACK,
            to_uri=to_uri,
            from_uri=SIPURI.parse(invite.from_header.split(";")[0].strip("<>")),
            call_id=invite.call_id,
            cseq=invite.cseq[0] if invite.cseq else 1,
        )

        # Copy To header with tag
        msg.headers["To"] = response.get_headers("To")

        return msg

    def build_bye(self, dialog: SIPDialog) -> SIPMessage:
        """Build BYE request."""
        return self.build_request(
            method=SIPMethod.BYE,
            to_uri=dialog.remote_target,
            from_uri=dialog.local_uri,
            call_id=dialog.call_id,
            cseq=dialog.get_next_cseq(),
        )

    def build_register(
        self,
        registrar_uri: SIPURI,
        aor: SIPURI,
        contact: SIPURI,
        expires: int = 3600,
    ) -> SIPMessage:
        """Build REGISTER request."""
        msg = self.build_request(
            method=SIPMethod.REGISTER,
            to_uri=registrar_uri,
            from_uri=aor,
        )

        msg.set_header("Contact", f"<{contact}>;expires={expires}")
        msg.set_header("Expires", str(expires))

        return msg


class SIPClient:
    """
    SIP client.

    Handles:
    - Making outbound calls
    - Receiving inbound calls
    - Registration
    - Message sending
    """

    def __init__(self, config: Optional[SIPConfig] = None):
        self.config = config or SIPConfig()
        self._builder = SIPMessageBuilder(self.config)

        # State
        self._sessions: Dict[str, SIPSession] = {}
        self._dialogs: Dict[str, SIPDialog] = {}
        self._transactions: Dict[str, SIPTransaction] = {}

        # Transport
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[asyncio.DatagramProtocol] = None

        # Callbacks
        self._on_invite: List[Callable[[SIPSession], Awaitable[None]]] = []
        self._on_bye: List[Callable[[SIPSession], Awaitable[None]]] = []
        self._on_message: List[Callable[[SIPMessage], Awaitable[None]]] = []

        self._running = False

    async def start(self) -> None:
        """Start SIP client."""
        if self._running:
            return

        self._running = True

        # Create UDP transport
        loop = asyncio.get_event_loop()

        class SIPProtocol(asyncio.DatagramProtocol):
            def __init__(self, client):
                self.client = client

            def datagram_received(self, data, addr):
                asyncio.create_task(self.client._handle_message(data, addr))

        transport, protocol = await loop.create_datagram_endpoint(
            lambda: SIPProtocol(self),
            local_addr=(self.config.local_ip, self.config.local_port),
        )

        self._transport = transport
        self._protocol = protocol

        logger.info(f"SIP client started on {self.config.local_ip}:{self.config.local_port}")

    async def stop(self) -> None:
        """Stop SIP client."""
        self._running = False

        if self._transport:
            self._transport.close()
            self._transport = None

        logger.info("SIP client stopped")

    async def call(
        self,
        to_uri: SIPURI,
        from_uri: SIPURI,
        sdp: str,
    ) -> SIPSession:
        """Make outbound call."""
        # Build INVITE
        invite = self._builder.build_invite(to_uri, from_uri, sdp)

        # Create dialog
        dialog = SIPDialog(
            dialog_id=str(uuid.uuid4()),
            call_id=invite.call_id,
            local_tag=invite.from_header.split("tag=")[1].split(";")[0],
            remote_tag="",
            local_uri=from_uri,
            remote_uri=to_uri,
            remote_target=to_uri,
        )

        # Create session
        session = SIPSession(
            session_id=str(uuid.uuid4()),
            dialog=dialog,
            local_sdp=sdp,
            direction="outbound",
            caller_uri=from_uri,
            callee_uri=to_uri,
        )

        self._sessions[session.session_id] = session
        self._dialogs[dialog.call_id] = dialog

        # Send INVITE
        await self._send_message(invite, (to_uri.host, to_uri.port))

        return session

    async def answer(
        self,
        session: SIPSession,
        sdp: str,
    ) -> None:
        """Answer incoming call."""
        session.local_sdp = sdp
        session.state = "connected"
        session.connected_at = datetime.utcnow()

        # Build 200 OK response
        # In production: send actual response

    async def hangup(self, session_id: str) -> bool:
        """Hang up call."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        # Build BYE
        bye = self._builder.build_bye(session.dialog)

        # Send BYE
        target = session.dialog.remote_target
        await self._send_message(bye, (target.host, target.port))

        # Update session
        session.state = "terminated"
        session.ended_at = datetime.utcnow()

        return True

    async def _send_message(
        self,
        message: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Send SIP message."""
        if self._transport:
            data = message.to_bytes()
            self._transport.sendto(data, addr)
            logger.debug(f"Sent SIP message to {addr}")

    async def _handle_message(
        self,
        data: bytes,
        addr: Tuple[str, int],
    ) -> None:
        """Handle received SIP message."""
        try:
            message = SIPMessage.parse(data)

            # Trigger callbacks
            for callback in self._on_message:
                await callback(message)

            if message.is_request:
                await self._handle_request(message, addr)
            else:
                await self._handle_response(message, addr)

        except Exception as e:
            logger.error(f"Error handling SIP message: {e}")

    async def _handle_request(
        self,
        request: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle SIP request."""
        if request.method == SIPMethod.INVITE:
            await self._handle_invite(request, addr)
        elif request.method == SIPMethod.BYE:
            await self._handle_bye(request, addr)
        elif request.method == SIPMethod.ACK:
            await self._handle_ack(request, addr)
        elif request.method == SIPMethod.CANCEL:
            await self._handle_cancel(request, addr)
        elif request.method == SIPMethod.OPTIONS:
            await self._handle_options(request, addr)

    async def _handle_response(
        self,
        response: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle SIP response."""
        call_id = response.call_id
        dialog = self._dialogs.get(call_id)

        if not dialog:
            return

        # Update dialog state based on response
        if response.status_code == 200:
            dialog.state = DialogState.CONFIRMED

            # Extract remote tag
            to_header = response.to_header
            if to_header and "tag=" in to_header:
                dialog.remote_tag = to_header.split("tag=")[1].split(";")[0]

            # Find session and update
            for session in self._sessions.values():
                if session.dialog.call_id == call_id:
                    session.state = "connected"
                    session.connected_at = datetime.utcnow()
                    session.remote_sdp = response.body
                    break

    async def _handle_invite(
        self,
        request: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle INVITE request."""
        # Send 100 Trying
        trying = self._builder.build_response(request, SIPStatus.TRYING)
        await self._send_message(trying, addr)

        # Create dialog
        from_header = request.from_header
        from_tag = from_header.split("tag=")[1].split(";")[0] if "tag=" in from_header else ""

        dialog = SIPDialog(
            dialog_id=str(uuid.uuid4()),
            call_id=request.call_id,
            local_tag=self._builder._generate_tag(),
            remote_tag=from_tag,
            remote_uri=request.request_uri,
            remote_target=request.request_uri,
        )

        self._dialogs[dialog.call_id] = dialog

        # Create session
        session = SIPSession(
            session_id=str(uuid.uuid4()),
            dialog=dialog,
            remote_sdp=request.body,
            direction="inbound",
            callee_uri=request.request_uri,
        )

        self._sessions[session.session_id] = session

        # Send 180 Ringing
        ringing = self._builder.build_response(request, SIPStatus.RINGING)
        await self._send_message(ringing, addr)

        # Trigger callbacks
        for callback in self._on_invite:
            await callback(session)

    async def _handle_bye(
        self,
        request: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle BYE request."""
        call_id = request.call_id

        # Find session
        session = None
        for s in self._sessions.values():
            if s.dialog.call_id == call_id:
                session = s
                break

        if session:
            session.state = "terminated"
            session.ended_at = datetime.utcnow()

            # Trigger callbacks
            for callback in self._on_bye:
                await callback(session)

        # Send 200 OK
        ok = self._builder.build_response(request, SIPStatus.OK)
        await self._send_message(ok, addr)

    async def _handle_ack(
        self,
        request: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle ACK request."""
        # ACK confirms dialog
        call_id = request.call_id
        dialog = self._dialogs.get(call_id)

        if dialog:
            dialog.state = DialogState.CONFIRMED

    async def _handle_cancel(
        self,
        request: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle CANCEL request."""
        # Send 200 OK for CANCEL
        ok = self._builder.build_response(request, SIPStatus.OK)
        await self._send_message(ok, addr)

        # Send 487 Request Terminated for original INVITE
        # In production: track and respond to original transaction

    async def _handle_options(
        self,
        request: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle OPTIONS request."""
        ok = self._builder.build_response(request, SIPStatus.OK)
        ok.set_header("Allow", "INVITE, ACK, BYE, CANCEL, OPTIONS")
        ok.set_header("Accept", "application/sdp")
        await self._send_message(ok, addr)

    def on_invite(self, callback: Callable[[SIPSession], Awaitable[None]]) -> None:
        """Register INVITE callback."""
        self._on_invite.append(callback)

    def on_bye(self, callback: Callable[[SIPSession], Awaitable[None]]) -> None:
        """Register BYE callback."""
        self._on_bye.append(callback)

    def on_message(self, callback: Callable[[SIPMessage], Awaitable[None]]) -> None:
        """Register message callback."""
        self._on_message.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "running": self._running,
            "active_sessions": len(self._sessions),
            "active_dialogs": len(self._dialogs),
        }


class SIPRegistrar:
    """
    SIP registrar.

    Handles:
    - User registration
    - Location service
    - Binding management
    """

    def __init__(self, config: Optional[SIPConfig] = None):
        self.config = config or SIPConfig()

        # Bindings: AOR -> List of contacts
        self._bindings: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        aor: str,
        contact: str,
        expires: int,
        call_id: str,
    ) -> bool:
        """Register contact for AOR."""
        async with self._lock:
            if aor not in self._bindings:
                self._bindings[aor] = []

            # Check for existing binding
            for binding in self._bindings[aor]:
                if binding["contact"] == contact:
                    binding["expires"] = datetime.utcnow() + timedelta(seconds=expires)
                    binding["call_id"] = call_id
                    return True

            # Add new binding
            self._bindings[aor].append({
                "contact": contact,
                "expires": datetime.utcnow() + timedelta(seconds=expires),
                "call_id": call_id,
                "registered_at": datetime.utcnow(),
            })

            return True

    async def unregister(self, aor: str, contact: Optional[str] = None) -> bool:
        """Unregister contact."""
        async with self._lock:
            if aor not in self._bindings:
                return False

            if contact:
                self._bindings[aor] = [
                    b for b in self._bindings[aor]
                    if b["contact"] != contact
                ]
            else:
                del self._bindings[aor]

            return True

    async def lookup(self, aor: str) -> List[str]:
        """Lookup contacts for AOR."""
        now = datetime.utcnow()

        async with self._lock:
            bindings = self._bindings.get(aor, [])

            # Filter expired bindings
            valid = [
                b["contact"] for b in bindings
                if b["expires"] > now
            ]

            return valid

    async def cleanup_expired(self) -> int:
        """Remove expired bindings."""
        now = datetime.utcnow()
        removed = 0

        async with self._lock:
            for aor in list(self._bindings.keys()):
                before = len(self._bindings[aor])
                self._bindings[aor] = [
                    b for b in self._bindings[aor]
                    if b["expires"] > now
                ]
                removed += before - len(self._bindings[aor])

                if not self._bindings[aor]:
                    del self._bindings[aor]

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get registrar statistics."""
        total_bindings = sum(len(b) for b in self._bindings.values())

        return {
            "total_aors": len(self._bindings),
            "total_bindings": total_bindings,
        }


class SIPProxy:
    """
    SIP proxy server.

    Handles:
    - Request routing
    - Load balancing
    - Failover
    """

    def __init__(
        self,
        config: Optional[SIPConfig] = None,
        registrar: Optional[SIPRegistrar] = None,
    ):
        self.config = config or SIPConfig()
        self.registrar = registrar or SIPRegistrar(config)

        # Routes
        self._routes: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()

    async def route(self, request: SIPMessage) -> List[Tuple[str, int]]:
        """Route SIP request to targets."""
        if not request.request_uri:
            return []

        # Try registrar first
        aor = str(request.request_uri)
        contacts = await self.registrar.lookup(aor)

        if contacts:
            targets = []
            for contact in contacts:
                uri = SIPURI.parse(contact)
                targets.append((uri.host, uri.port))
            return targets

        # Try static routes
        domain = request.request_uri.host
        route_targets = self._routes.get(domain, [])

        targets = []
        for target in route_targets:
            uri = SIPURI.parse(target)
            targets.append((uri.host, uri.port))

        return targets

    def add_route(self, domain: str, target: str) -> None:
        """Add routing rule."""
        if domain not in self._routes:
            self._routes[domain] = []
        self._routes[domain].append(target)

    def remove_route(self, domain: str, target: Optional[str] = None) -> None:
        """Remove routing rule."""
        if domain in self._routes:
            if target:
                self._routes[domain] = [
                    t for t in self._routes[domain] if t != target
                ]
            else:
                del self._routes[domain]

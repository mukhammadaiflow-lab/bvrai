"""
SIP Protocol Module

This module provides SIP (Session Initiation Protocol) support
for voice communications with PBX systems and SIP trunks.
"""

import asyncio
import hashlib
import logging
import random
import re
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from .base import (
    Call,
    CallDirection,
    CallEvent,
    CallEventType,
    CallState,
)


logger = logging.getLogger(__name__)


class SIPMethod(str, Enum):
    """SIP request methods."""
    INVITE = "INVITE"
    ACK = "ACK"
    BYE = "BYE"
    CANCEL = "CANCEL"
    REGISTER = "REGISTER"
    OPTIONS = "OPTIONS"
    INFO = "INFO"
    UPDATE = "UPDATE"
    REFER = "REFER"
    NOTIFY = "NOTIFY"
    SUBSCRIBE = "SUBSCRIBE"
    MESSAGE = "MESSAGE"


class SIPStatus(int, Enum):
    """Common SIP response codes."""
    # 1xx Provisional
    TRYING = 100
    RINGING = 180
    SESSION_PROGRESS = 183

    # 2xx Success
    OK = 200

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
    """SIP dialog state."""
    NULL = "null"
    EARLY = "early"
    CONFIRMED = "confirmed"
    TERMINATED = "terminated"


@dataclass
class SIPAddress:
    """SIP address (URI)."""

    user: str = ""
    host: str = ""
    port: int = 5060
    display_name: Optional[str] = None
    transport: str = "udp"
    params: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        """Convert to SIP URI string."""
        uri = f"sip:{self.user}@{self.host}"
        if self.port != 5060:
            uri += f":{self.port}"

        if self.transport != "udp":
            uri += f";transport={self.transport}"

        for key, value in self.params.items():
            uri += f";{key}={value}"

        if self.display_name:
            return f'"{self.display_name}" <{uri}>'

        return f"<{uri}>"

    @classmethod
    def parse(cls, uri: str) -> "SIPAddress":
        """Parse SIP URI."""
        address = cls()

        # Extract display name if present
        display_match = re.match(r'"([^"]+)"\s*<(.+)>', uri)
        if display_match:
            address.display_name = display_match.group(1)
            uri = display_match.group(2)
        else:
            # Handle bare <uri>
            uri = uri.strip("<>")

        # Remove sip: prefix
        if uri.startswith("sip:"):
            uri = uri[4:]

        # Split user@host
        if "@" in uri:
            user_part, host_part = uri.split("@", 1)
            address.user = user_part
        else:
            host_part = uri

        # Parse host:port;params
        if ";" in host_part:
            host_port, params = host_part.split(";", 1)
            for param in params.split(";"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    if key == "transport":
                        address.transport = value
                    else:
                        address.params[key] = value
        else:
            host_port = host_part

        # Parse host:port
        if ":" in host_port:
            address.host, port_str = host_port.split(":", 1)
            address.port = int(port_str)
        else:
            address.host = host_port

        return address


@dataclass
class SIPHeader:
    """SIP message header."""

    name: str
    value: str
    params: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        """Convert to header line."""
        result = f"{self.name}: {self.value}"
        for key, value in self.params.items():
            result += f";{key}={value}"
        return result


@dataclass
class SIPMessage:
    """SIP message (request or response)."""

    # Request line (for requests)
    method: Optional[SIPMethod] = None
    request_uri: Optional[str] = None

    # Status line (for responses)
    status_code: Optional[int] = None
    reason_phrase: Optional[str] = None

    # Headers
    headers: Dict[str, SIPHeader] = field(default_factory=dict)

    # Body
    body: str = ""

    # Metadata
    is_request: bool = True

    def get_header(self, name: str) -> Optional[str]:
        """Get header value by name."""
        header = self.headers.get(name.lower())
        return header.value if header else None

    def set_header(self, name: str, value: str, **params) -> None:
        """Set header value."""
        self.headers[name.lower()] = SIPHeader(name=name, value=value, params=params)

    @property
    def call_id(self) -> Optional[str]:
        """Get Call-ID header."""
        return self.get_header("Call-ID") or self.get_header("i")

    @property
    def from_header(self) -> Optional[str]:
        """Get From header."""
        return self.get_header("From") or self.get_header("f")

    @property
    def to_header(self) -> Optional[str]:
        """Get To header."""
        return self.get_header("To") or self.get_header("t")

    @property
    def cseq(self) -> Optional[Tuple[int, str]]:
        """Get CSeq as (sequence, method)."""
        value = self.get_header("CSeq")
        if value:
            parts = value.split()
            if len(parts) == 2:
                return (int(parts[0]), parts[1])
        return None

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        lines = []

        if self.is_request:
            lines.append(f"{self.method.value} {self.request_uri} SIP/2.0")
        else:
            lines.append(f"SIP/2.0 {self.status_code} {self.reason_phrase}")

        # Add headers
        for header in self.headers.values():
            lines.append(str(header))

        # Empty line before body
        lines.append("")

        # Body
        if self.body:
            lines.append(self.body)

        return "\r\n".join(lines).encode("utf-8")

    @classmethod
    def parse(cls, data: bytes) -> "SIPMessage":
        """Parse SIP message from bytes."""
        text = data.decode("utf-8")
        lines = text.split("\r\n")

        message = cls()

        # Parse first line
        first_line = lines[0]
        if first_line.startswith("SIP/2.0"):
            # Response
            message.is_request = False
            parts = first_line.split(" ", 2)
            message.status_code = int(parts[1])
            message.reason_phrase = parts[2] if len(parts) > 2 else ""
        else:
            # Request
            message.is_request = True
            parts = first_line.split(" ")
            message.method = SIPMethod(parts[0])
            message.request_uri = parts[1]

        # Parse headers
        i = 1
        while i < len(lines) and lines[i]:
            line = lines[i]
            if ":" in line:
                name, value = line.split(":", 1)
                message.set_header(name.strip(), value.strip())
            i += 1

        # Body is everything after empty line
        if i < len(lines):
            message.body = "\r\n".join(lines[i + 1:])

        return message


@dataclass
class SIPDialog:
    """SIP dialog (call leg)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str = ""

    # Dialog identifiers
    local_tag: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    remote_tag: Optional[str] = None

    # Addresses
    local_uri: Optional[SIPAddress] = None
    remote_uri: Optional[SIPAddress] = None
    remote_target: Optional[str] = None

    # Route set
    route_set: List[str] = field(default_factory=list)

    # State
    state: DialogState = DialogState.NULL

    # Sequence numbers
    local_cseq: int = field(default_factory=lambda: random.randint(1, 2**31 - 1))
    remote_cseq: Optional[int] = None

    # Associated call
    call: Optional[Call] = None

    def get_next_cseq(self) -> int:
        """Get and increment local CSeq."""
        seq = self.local_cseq
        self.local_cseq += 1
        return seq


@dataclass
class SIPSession:
    """SIP session with media."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dialog: SIPDialog = field(default_factory=SIPDialog)

    # SDP
    local_sdp: Optional[str] = None
    remote_sdp: Optional[str] = None

    # Media endpoints
    local_rtp_port: int = 0
    local_rtp_host: str = ""
    remote_rtp_port: int = 0
    remote_rtp_host: str = ""

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    established_at: Optional[datetime] = None
    terminated_at: Optional[datetime] = None


@dataclass
class SIPConfig:
    """SIP provider configuration."""

    # Identity
    username: str = ""
    password: str = ""
    domain: str = ""
    display_name: Optional[str] = None

    # Server
    registrar_host: str = ""
    registrar_port: int = 5060
    proxy_host: Optional[str] = None
    proxy_port: int = 5060
    outbound_proxy: Optional[str] = None

    # Transport
    transport: str = "udp"  # udp, tcp, tls

    # Local binding
    local_host: str = "0.0.0.0"
    local_port: int = 5060

    # Registration
    register: bool = True
    registration_expires: int = 3600

    # NAT
    use_stun: bool = False
    stun_server: Optional[str] = None

    # Timeouts
    transaction_timeout: int = 32  # seconds
    dialog_timeout: int = 1800

    # Codecs
    supported_codecs: List[str] = field(default_factory=lambda: ["PCMU", "PCMA", "opus"])


class SIPProvider:
    """
    SIP telephony provider.

    Handles SIP signaling, registration, and call management.
    """

    def __init__(self, config: SIPConfig):
        """
        Initialize SIP provider.

        Args:
            config: SIP configuration
        """
        self.config = config
        self._transport: Optional[asyncio.DatagramProtocol] = None
        self._dialogs: Dict[str, SIPDialog] = {}
        self._sessions: Dict[str, SIPSession] = {}
        self._transactions: Dict[str, Any] = {}
        self._registered: bool = False
        self._call_handlers: Dict[str, Callable] = {}

    async def start(self) -> None:
        """Start the SIP provider."""
        loop = asyncio.get_event_loop()

        # Create UDP transport
        if self.config.transport == "udp":
            _, self._transport = await loop.create_datagram_endpoint(
                lambda: SIPTransport(self),
                local_addr=(self.config.local_host, self.config.local_port),
            )

        # Register if configured
        if self.config.register:
            await self.register()

        logger.info(f"SIP provider started on {self.config.local_host}:{self.config.local_port}")

    async def stop(self) -> None:
        """Stop the SIP provider."""
        # Unregister
        if self._registered:
            await self.unregister()

        # Close transport
        if self._transport:
            self._transport.close()

        logger.info("SIP provider stopped")

    async def register(self) -> bool:
        """
        Register with SIP registrar.

        Returns:
            True if registration successful
        """
        # Build REGISTER request
        request_uri = f"sip:{self.config.domain}"
        from_addr = SIPAddress(
            user=self.config.username,
            host=self.config.domain,
            display_name=self.config.display_name,
        )
        to_addr = from_addr

        message = SIPMessage(
            method=SIPMethod.REGISTER,
            request_uri=request_uri,
            is_request=True,
        )

        call_id = str(uuid.uuid4())
        cseq = random.randint(1, 2**31 - 1)

        message.set_header("Via", f"SIP/2.0/UDP {self.config.local_host}:{self.config.local_port};branch=z9hG4bK{uuid.uuid4().hex[:8]}")
        message.set_header("From", str(from_addr), tag=str(uuid.uuid4())[:8])
        message.set_header("To", str(to_addr))
        message.set_header("Call-ID", call_id)
        message.set_header("CSeq", f"{cseq} REGISTER")
        message.set_header("Contact", f"<sip:{self.config.username}@{self.config.local_host}:{self.config.local_port}>")
        message.set_header("Expires", str(self.config.registration_expires))
        message.set_header("Content-Length", "0")

        # Send request
        await self._send_message(
            message,
            (self.config.registrar_host, self.config.registrar_port),
        )

        # TODO: Handle response, authentication
        self._registered = True

        return True

    async def unregister(self) -> None:
        """Unregister from SIP registrar."""
        self._registered = False

    async def make_call(
        self,
        to_number: str,
        from_number: Optional[str] = None,
    ) -> Call:
        """
        Make an outbound call.

        Args:
            to_number: Number to call
            from_number: Caller ID

        Returns:
            Call object
        """
        from_number = from_number or self.config.username

        # Create call object
        call = Call(
            direction=CallDirection.OUTBOUND,
            state=CallState.INITIATED,
            from_number=from_number,
            to_number=to_number,
            provider="sip",
        )

        # Create dialog
        dialog = SIPDialog(
            call_id=str(uuid.uuid4()),
            local_uri=SIPAddress(user=from_number, host=self.config.domain),
            remote_uri=SIPAddress(user=to_number, host=self.config.domain),
            call=call,
        )

        call.provider_call_id = dialog.call_id
        self._dialogs[dialog.call_id] = dialog

        # Create session
        session = SIPSession(dialog=dialog)
        session.local_sdp = self._build_sdp()
        self._sessions[session.id] = session

        # Build INVITE
        request_uri = str(dialog.remote_uri).strip("<>")
        message = SIPMessage(
            method=SIPMethod.INVITE,
            request_uri=request_uri,
            is_request=True,
        )

        cseq = dialog.get_next_cseq()

        message.set_header("Via", f"SIP/2.0/UDP {self.config.local_host}:{self.config.local_port};branch=z9hG4bK{uuid.uuid4().hex[:8]}")
        message.set_header("From", str(dialog.local_uri), tag=dialog.local_tag)
        message.set_header("To", str(dialog.remote_uri))
        message.set_header("Call-ID", dialog.call_id)
        message.set_header("CSeq", f"{cseq} INVITE")
        message.set_header("Contact", f"<sip:{self.config.username}@{self.config.local_host}:{self.config.local_port}>")
        message.set_header("Content-Type", "application/sdp")
        message.set_header("Content-Length", str(len(session.local_sdp)))
        message.body = session.local_sdp

        # Send INVITE
        target = self.config.proxy_host or self.config.domain
        target_port = self.config.proxy_port

        await self._send_message(message, (target, target_port))

        dialog.state = DialogState.EARLY
        call.add_event(CallEventType.INITIATED)

        logger.info(f"SIP INVITE sent for call {call.id}")

        return call

    async def answer_call(self, call_id: str) -> None:
        """
        Answer an incoming call.

        Args:
            call_id: Call ID (SIP Call-ID)
        """
        dialog = self._dialogs.get(call_id)
        if not dialog or not dialog.call:
            return

        # Build 200 OK response
        message = SIPMessage(
            is_request=False,
            status_code=200,
            reason_phrase="OK",
        )

        # TODO: Build proper response with SDP

        dialog.state = DialogState.CONFIRMED
        dialog.call.state = CallState.IN_PROGRESS
        dialog.call.answered_at = datetime.utcnow()
        dialog.call.add_event(CallEventType.ANSWERED)

    async def end_call(self, call_id: str) -> None:
        """
        End a call.

        Args:
            call_id: Call ID
        """
        dialog = self._dialogs.get(call_id)
        if not dialog:
            return

        # Build BYE request
        request_uri = dialog.remote_target or str(dialog.remote_uri).strip("<>")
        message = SIPMessage(
            method=SIPMethod.BYE,
            request_uri=request_uri,
            is_request=True,
        )

        cseq = dialog.get_next_cseq()

        message.set_header("Via", f"SIP/2.0/UDP {self.config.local_host}:{self.config.local_port};branch=z9hG4bK{uuid.uuid4().hex[:8]}")
        message.set_header("From", str(dialog.local_uri), tag=dialog.local_tag)
        message.set_header("To", str(dialog.remote_uri), tag=dialog.remote_tag or "")
        message.set_header("Call-ID", dialog.call_id)
        message.set_header("CSeq", f"{cseq} BYE")
        message.set_header("Content-Length", "0")

        target = self.config.proxy_host or dialog.remote_uri.host
        target_port = self.config.proxy_port

        await self._send_message(message, (target, target_port))

        dialog.state = DialogState.TERMINATED
        if dialog.call:
            dialog.call.state = CallState.COMPLETED
            dialog.call.ended_at = datetime.utcnow()
            dialog.call.add_event(CallEventType.COMPLETED)

        logger.info(f"SIP BYE sent for call {call_id}")

    async def handle_message(
        self,
        message: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """
        Handle received SIP message.

        Args:
            message: SIP message
            addr: Source address
        """
        if message.is_request:
            await self._handle_request(message, addr)
        else:
            await self._handle_response(message, addr)

    async def _handle_request(
        self,
        message: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle incoming SIP request."""
        method = message.method

        if method == SIPMethod.INVITE:
            await self._handle_invite(message, addr)
        elif method == SIPMethod.ACK:
            await self._handle_ack(message, addr)
        elif method == SIPMethod.BYE:
            await self._handle_bye(message, addr)
        elif method == SIPMethod.CANCEL:
            await self._handle_cancel(message, addr)
        elif method == SIPMethod.OPTIONS:
            await self._handle_options(message, addr)

    async def _handle_response(
        self,
        message: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle incoming SIP response."""
        call_id = message.call_id
        dialog = self._dialogs.get(call_id)

        if not dialog:
            logger.warning(f"Response for unknown dialog: {call_id}")
            return

        status = message.status_code

        if status == 180:  # Ringing
            dialog.state = DialogState.EARLY
            if dialog.call:
                dialog.call.state = CallState.RINGING
                dialog.call.add_event(CallEventType.RINGING)

        elif status == 200:  # OK
            dialog.state = DialogState.CONFIRMED
            if dialog.call:
                dialog.call.state = CallState.IN_PROGRESS
                dialog.call.answered_at = datetime.utcnow()
                dialog.call.add_event(CallEventType.ANSWERED)

            # Extract remote tag
            to_header = message.get_header("To")
            if to_header and "tag=" in to_header:
                tag_match = re.search(r'tag=([^\s;]+)', to_header)
                if tag_match:
                    dialog.remote_tag = tag_match.group(1)

            # Send ACK
            await self._send_ack(dialog, addr)

        elif 400 <= status < 700:  # Error responses
            dialog.state = DialogState.TERMINATED
            if dialog.call:
                dialog.call.state = CallState.FAILED
                dialog.call.add_event(
                    CallEventType.FAILED,
                    {"status": status, "reason": message.reason_phrase}
                )

    async def _handle_invite(
        self,
        message: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle incoming INVITE."""
        call_id = message.call_id

        # Create new dialog for incoming call
        dialog = SIPDialog(
            call_id=call_id,
        )

        # Parse From and To
        from_header = message.get_header("From")
        to_header = message.get_header("To")

        if from_header:
            dialog.remote_uri = SIPAddress.parse(from_header)
            # Extract tag
            if "tag=" in from_header:
                tag_match = re.search(r'tag=([^\s;]+)', from_header)
                if tag_match:
                    dialog.remote_tag = tag_match.group(1)

        if to_header:
            dialog.local_uri = SIPAddress.parse(to_header)

        # Create call
        call = Call(
            direction=CallDirection.INBOUND,
            state=CallState.RINGING,
            from_number=dialog.remote_uri.user if dialog.remote_uri else "",
            to_number=dialog.local_uri.user if dialog.local_uri else "",
            provider="sip",
            provider_call_id=call_id,
        )

        dialog.call = call
        self._dialogs[call_id] = dialog

        # Send 180 Ringing
        ringing = SIPMessage(
            is_request=False,
            status_code=180,
            reason_phrase="Ringing",
        )

        # Copy required headers
        ringing.set_header("Via", message.get_header("Via") or "")
        ringing.set_header("From", from_header or "")
        ringing.set_header("To", f"{to_header};tag={dialog.local_tag}" if to_header else "")
        ringing.set_header("Call-ID", call_id)
        ringing.set_header("CSeq", message.get_header("CSeq") or "")
        ringing.set_header("Content-Length", "0")

        await self._send_message(ringing, addr)

        call.add_event(CallEventType.RINGING)

        # Notify handler
        handler = self._call_handlers.get("incoming")
        if handler:
            await handler(call)

    async def _handle_ack(
        self,
        message: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle ACK."""
        call_id = message.call_id
        dialog = self._dialogs.get(call_id)

        if dialog:
            dialog.state = DialogState.CONFIRMED
            logger.debug(f"ACK received for dialog {call_id}")

    async def _handle_bye(
        self,
        message: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle BYE."""
        call_id = message.call_id
        dialog = self._dialogs.get(call_id)

        if dialog:
            dialog.state = DialogState.TERMINATED
            if dialog.call:
                dialog.call.state = CallState.COMPLETED
                dialog.call.ended_at = datetime.utcnow()
                dialog.call.hangup_source = "caller"
                dialog.call.add_event(CallEventType.COMPLETED)

            # Send 200 OK
            ok = SIPMessage(
                is_request=False,
                status_code=200,
                reason_phrase="OK",
            )

            ok.set_header("Via", message.get_header("Via") or "")
            ok.set_header("From", message.get_header("From") or "")
            ok.set_header("To", message.get_header("To") or "")
            ok.set_header("Call-ID", call_id)
            ok.set_header("CSeq", message.get_header("CSeq") or "")
            ok.set_header("Content-Length", "0")

            await self._send_message(ok, addr)

            logger.info(f"Call ended by BYE: {call_id}")

    async def _handle_cancel(
        self,
        message: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle CANCEL."""
        call_id = message.call_id
        dialog = self._dialogs.get(call_id)

        if dialog:
            dialog.state = DialogState.TERMINATED
            if dialog.call:
                dialog.call.state = CallState.CANCELED
                dialog.call.add_event(CallEventType.FAILED, {"reason": "canceled"})

    async def _handle_options(
        self,
        message: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Handle OPTIONS (keepalive/ping)."""
        # Send 200 OK
        ok = SIPMessage(
            is_request=False,
            status_code=200,
            reason_phrase="OK",
        )

        ok.set_header("Via", message.get_header("Via") or "")
        ok.set_header("From", message.get_header("From") or "")
        ok.set_header("To", message.get_header("To") or "")
        ok.set_header("Call-ID", message.call_id or "")
        ok.set_header("CSeq", message.get_header("CSeq") or "")
        ok.set_header("Allow", "INVITE, ACK, BYE, CANCEL, OPTIONS")
        ok.set_header("Content-Length", "0")

        await self._send_message(ok, addr)

    async def _send_ack(
        self,
        dialog: SIPDialog,
        addr: Tuple[str, int],
    ) -> None:
        """Send ACK for a dialog."""
        request_uri = dialog.remote_target or str(dialog.remote_uri).strip("<>")

        ack = SIPMessage(
            method=SIPMethod.ACK,
            request_uri=request_uri,
            is_request=True,
        )

        cseq = dialog.local_cseq - 1  # ACK uses same CSeq as INVITE

        ack.set_header("Via", f"SIP/2.0/UDP {self.config.local_host}:{self.config.local_port};branch=z9hG4bK{uuid.uuid4().hex[:8]}")
        ack.set_header("From", str(dialog.local_uri), tag=dialog.local_tag)
        ack.set_header("To", str(dialog.remote_uri), tag=dialog.remote_tag or "")
        ack.set_header("Call-ID", dialog.call_id)
        ack.set_header("CSeq", f"{cseq} ACK")
        ack.set_header("Content-Length", "0")

        await self._send_message(ack, addr)

    async def _send_message(
        self,
        message: SIPMessage,
        addr: Tuple[str, int],
    ) -> None:
        """Send SIP message."""
        if self._transport:
            data = message.to_bytes()
            self._transport.sendto(data, addr)

    def _build_sdp(self) -> str:
        """Build SDP for outgoing calls."""
        # Allocate RTP port (simplified)
        rtp_port = 10000 + random.randint(0, 10000)

        lines = [
            "v=0",
            f"o=- {random.randint(1, 2**32)} 1 IN IP4 {self.config.local_host}",
            "s=VoiceSession",
            f"c=IN IP4 {self.config.local_host}",
            "t=0 0",
            f"m=audio {rtp_port} RTP/AVP 0 8 101",
            "a=rtpmap:0 PCMU/8000",
            "a=rtpmap:8 PCMA/8000",
            "a=rtpmap:101 telephone-event/8000",
            "a=fmtp:101 0-16",
            "a=sendrecv",
        ]

        return "\r\n".join(lines) + "\r\n"

    def register_handler(self, event: str, handler: Callable) -> None:
        """Register event handler."""
        self._call_handlers[event] = handler


class SIPTransport(asyncio.DatagramProtocol):
    """UDP transport for SIP messages."""

    def __init__(self, provider: SIPProvider):
        self.provider = provider

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        try:
            message = SIPMessage.parse(data)
            asyncio.create_task(self.provider.handle_message(message, addr))
        except Exception as e:
            logger.error(f"Failed to parse SIP message: {e}")

    def error_received(self, exc):
        logger.error(f"SIP transport error: {exc}")


__all__ = [
    # Enums
    "SIPMethod",
    "SIPStatus",
    "DialogState",
    # Data classes
    "SIPAddress",
    "SIPHeader",
    "SIPMessage",
    "SIPDialog",
    "SIPSession",
    "SIPConfig",
    # Classes
    "SIPProvider",
]

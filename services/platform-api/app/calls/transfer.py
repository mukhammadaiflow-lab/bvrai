"""
Call Transfer Handler

Call transfer capabilities:
- Blind transfer
- Attended transfer
- Warm transfer
- Conference bridge
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)


class TransferType(str, Enum):
    """Types of call transfer."""
    BLIND = "blind"           # Transfer without consultation
    ATTENDED = "attended"     # Transfer with consultation
    WARM = "warm"             # Warm handoff with context
    CONFERENCE = "conference" # Add party to call


class TransferStatus(str, Enum):
    """Transfer operation status."""
    INITIATED = "initiated"
    RINGING = "ringing"
    CONNECTED = "connected"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TransferReason(str, Enum):
    """Reason for transfer."""
    AGENT_REQUEST = "agent_request"
    CUSTOMER_REQUEST = "customer_request"
    ESCALATION = "escalation"
    SKILL_MISMATCH = "skill_mismatch"
    QUEUE_OVERFLOW = "queue_overflow"
    AFTER_HOURS = "after_hours"
    IVR_ROUTING = "ivr_routing"


@dataclass
class TransferRequest:
    """Transfer request."""
    transfer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str = ""
    transfer_type: TransferType = TransferType.BLIND

    # Source
    from_agent_id: Optional[str] = None
    from_queue_id: Optional[str] = None

    # Target
    to_agent_id: Optional[str] = None
    to_queue_id: Optional[str] = None
    to_number: Optional[str] = None
    to_sip_uri: Optional[str] = None

    # Options
    reason: TransferReason = TransferReason.AGENT_REQUEST
    announce_caller: bool = True
    preserve_context: bool = True
    timeout_seconds: int = 30

    # Context to transfer
    context: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    # Timestamps
    requested_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def target(self) -> str:
        """Get transfer target identifier."""
        if self.to_agent_id:
            return f"agent:{self.to_agent_id}"
        if self.to_queue_id:
            return f"queue:{self.to_queue_id}"
        if self.to_number:
            return f"number:{self.to_number}"
        if self.to_sip_uri:
            return f"sip:{self.to_sip_uri}"
        return "unknown"


@dataclass
class TransferResult:
    """Result of transfer operation."""
    transfer_id: str
    success: bool
    status: TransferStatus
    transfer_type: TransferType

    # Details
    error: Optional[str] = None
    target_call_id: Optional[str] = None
    connected_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Duration
    ring_duration_seconds: float = 0.0
    consultation_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0


@dataclass
class TransferState:
    """State of an active transfer."""
    request: TransferRequest
    status: TransferStatus = TransferStatus.INITIATED

    # Parties
    original_call_id: str = ""
    consultation_call_id: Optional[str] = None
    conference_id: Optional[str] = None

    # Timing
    initiated_at: datetime = field(default_factory=datetime.utcnow)
    ring_started_at: Optional[datetime] = None
    connected_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Audio state
    original_on_hold: bool = False
    transfer_party_connected: bool = False


class TransferHandler(ABC):
    """Abstract transfer handler."""

    @abstractmethod
    async def initiate(self, request: TransferRequest) -> TransferState:
        """Initiate transfer."""
        pass

    @abstractmethod
    async def complete(self, state: TransferState) -> TransferResult:
        """Complete transfer."""
        pass

    @abstractmethod
    async def cancel(self, state: TransferState) -> TransferResult:
        """Cancel transfer."""
        pass


class BlindTransfer(TransferHandler):
    """
    Blind (cold) transfer.

    Transfers the call immediately without consultation.
    The original agent is disconnected as soon as transfer starts.
    """

    def __init__(self):
        self._active_transfers: Dict[str, TransferState] = {}

    async def initiate(self, request: TransferRequest) -> TransferState:
        """Initiate blind transfer."""
        state = TransferState(
            request=request,
            status=TransferStatus.INITIATED,
            original_call_id=request.call_id,
        )

        self._active_transfers[request.transfer_id] = state

        logger.info(
            f"Blind transfer initiated: {request.call_id} -> {request.target}"
        )

        try:
            # In production: initiate transfer via telephony provider
            # 1. Put original party on hold with transfer music
            # 2. Dial transfer target
            # 3. On answer, bridge calls
            # 4. Disconnect originating agent

            state.status = TransferStatus.RINGING
            state.ring_started_at = datetime.utcnow()

            # Simulate dial delay
            await asyncio.sleep(0.1)

            # Simulate successful connection
            state.status = TransferStatus.CONNECTED
            state.connected_at = datetime.utcnow()
            state.transfer_party_connected = True

            return state

        except Exception as e:
            state.status = TransferStatus.FAILED
            logger.error(f"Blind transfer failed: {e}")
            return state

    async def complete(self, state: TransferState) -> TransferResult:
        """Complete blind transfer."""
        if state.status != TransferStatus.CONNECTED:
            return TransferResult(
                transfer_id=state.request.transfer_id,
                success=False,
                status=state.status,
                transfer_type=TransferType.BLIND,
                error="Transfer not connected",
            )

        state.status = TransferStatus.COMPLETED
        state.completed_at = datetime.utcnow()

        # Calculate durations
        ring_duration = 0.0
        if state.ring_started_at and state.connected_at:
            ring_duration = (state.connected_at - state.ring_started_at).total_seconds()

        total_duration = (state.completed_at - state.initiated_at).total_seconds()

        # Cleanup
        self._active_transfers.pop(state.request.transfer_id, None)

        return TransferResult(
            transfer_id=state.request.transfer_id,
            success=True,
            status=TransferStatus.COMPLETED,
            transfer_type=TransferType.BLIND,
            connected_at=state.connected_at,
            completed_at=state.completed_at,
            ring_duration_seconds=ring_duration,
            total_duration_seconds=total_duration,
        )

    async def cancel(self, state: TransferState) -> TransferResult:
        """Cancel blind transfer."""
        state.status = TransferStatus.CANCELLED
        state.completed_at = datetime.utcnow()

        self._active_transfers.pop(state.request.transfer_id, None)

        return TransferResult(
            transfer_id=state.request.transfer_id,
            success=False,
            status=TransferStatus.CANCELLED,
            transfer_type=TransferType.BLIND,
            completed_at=state.completed_at,
        )


class AttendedTransfer(TransferHandler):
    """
    Attended (consultative) transfer.

    Allows the agent to speak with the transfer target
    before completing the transfer.
    """

    def __init__(self):
        self._active_transfers: Dict[str, TransferState] = {}

    async def initiate(self, request: TransferRequest) -> TransferState:
        """Initiate attended transfer."""
        state = TransferState(
            request=request,
            status=TransferStatus.INITIATED,
            original_call_id=request.call_id,
        )

        self._active_transfers[request.transfer_id] = state

        logger.info(
            f"Attended transfer initiated: {request.call_id} -> {request.target}"
        )

        try:
            # In production:
            # 1. Put customer on hold
            # 2. Create consultation call to target
            # 3. Connect agent to consultation call
            # 4. Agent can then complete or cancel transfer

            state.original_on_hold = True
            state.status = TransferStatus.RINGING
            state.ring_started_at = datetime.utcnow()

            # Create consultation call ID
            state.consultation_call_id = f"consult-{uuid.uuid4()}"

            # Simulate dial
            await asyncio.sleep(0.1)

            state.status = TransferStatus.CONNECTED
            state.connected_at = datetime.utcnow()
            state.transfer_party_connected = True

            return state

        except Exception as e:
            state.status = TransferStatus.FAILED
            logger.error(f"Attended transfer failed: {e}")
            return state

    async def complete(self, state: TransferState) -> TransferResult:
        """Complete attended transfer (connect customer to target)."""
        if state.status != TransferStatus.CONNECTED:
            return TransferResult(
                transfer_id=state.request.transfer_id,
                success=False,
                status=state.status,
                transfer_type=TransferType.ATTENDED,
                error="Consultation not connected",
            )

        # In production:
        # 1. Bridge customer to transfer target
        # 2. Disconnect original agent
        # 3. End consultation leg

        state.status = TransferStatus.COMPLETED
        state.completed_at = datetime.utcnow()

        # Calculate durations
        ring_duration = 0.0
        consultation_duration = 0.0

        if state.ring_started_at and state.connected_at:
            ring_duration = (state.connected_at - state.ring_started_at).total_seconds()

        if state.connected_at and state.completed_at:
            consultation_duration = (state.completed_at - state.connected_at).total_seconds()

        total_duration = (state.completed_at - state.initiated_at).total_seconds()

        self._active_transfers.pop(state.request.transfer_id, None)

        return TransferResult(
            transfer_id=state.request.transfer_id,
            success=True,
            status=TransferStatus.COMPLETED,
            transfer_type=TransferType.ATTENDED,
            target_call_id=state.consultation_call_id,
            connected_at=state.connected_at,
            completed_at=state.completed_at,
            ring_duration_seconds=ring_duration,
            consultation_duration_seconds=consultation_duration,
            total_duration_seconds=total_duration,
        )

    async def cancel(self, state: TransferState) -> TransferResult:
        """Cancel attended transfer (return to customer)."""
        # In production:
        # 1. End consultation call
        # 2. Reconnect agent to customer
        # 3. Take customer off hold

        state.status = TransferStatus.CANCELLED
        state.completed_at = datetime.utcnow()
        state.original_on_hold = False

        self._active_transfers.pop(state.request.transfer_id, None)

        logger.info(f"Attended transfer cancelled, returning to customer")

        return TransferResult(
            transfer_id=state.request.transfer_id,
            success=False,
            status=TransferStatus.CANCELLED,
            transfer_type=TransferType.ATTENDED,
            completed_at=state.completed_at,
        )

    async def swap(self, state: TransferState) -> bool:
        """
        Swap between customer and consultation call.

        Used during attended transfer to toggle between parties.
        """
        if state.status != TransferStatus.CONNECTED:
            return False

        # Toggle hold state
        state.original_on_hold = not state.original_on_hold

        logger.debug(
            f"Swapped calls, customer on hold: {state.original_on_hold}"
        )

        return True


class WarmTransfer(TransferHandler):
    """
    Warm transfer with context.

    Similar to attended transfer but includes:
    - Context sharing
    - Introduction/announcement
    - Seamless handoff
    """

    def __init__(self):
        self._active_transfers: Dict[str, TransferState] = {}

    async def initiate(self, request: TransferRequest) -> TransferState:
        """Initiate warm transfer."""
        state = TransferState(
            request=request,
            status=TransferStatus.INITIATED,
            original_call_id=request.call_id,
        )

        self._active_transfers[request.transfer_id] = state

        logger.info(
            f"Warm transfer initiated: {request.call_id} -> {request.target}"
        )

        try:
            # Put customer on hold with context message
            state.original_on_hold = True
            state.status = TransferStatus.RINGING
            state.ring_started_at = datetime.utcnow()

            # Create consultation call
            state.consultation_call_id = f"warm-{uuid.uuid4()}"

            # Simulate dial
            await asyncio.sleep(0.1)

            # Send context to target (in production: via out-of-band signaling)
            if request.preserve_context:
                logger.info(f"Sharing context with transfer target: {request.context}")

            state.status = TransferStatus.CONNECTED
            state.connected_at = datetime.utcnow()
            state.transfer_party_connected = True

            return state

        except Exception as e:
            state.status = TransferStatus.FAILED
            logger.error(f"Warm transfer failed: {e}")
            return state

    async def complete(self, state: TransferState) -> TransferResult:
        """Complete warm transfer with introduction."""
        if state.status != TransferStatus.CONNECTED:
            return TransferResult(
                transfer_id=state.request.transfer_id,
                success=False,
                status=state.status,
                transfer_type=TransferType.WARM,
                error="Consultation not connected",
            )

        # In production:
        # 1. Create 3-way conference briefly for introduction
        # 2. Allow agent to introduce
        # 3. Drop original agent
        # 4. Continue call between customer and target

        state.status = TransferStatus.COMPLETED
        state.completed_at = datetime.utcnow()

        # Calculate durations
        ring_duration = 0.0
        consultation_duration = 0.0

        if state.ring_started_at and state.connected_at:
            ring_duration = (state.connected_at - state.ring_started_at).total_seconds()

        if state.connected_at and state.completed_at:
            consultation_duration = (state.completed_at - state.connected_at).total_seconds()

        total_duration = (state.completed_at - state.initiated_at).total_seconds()

        self._active_transfers.pop(state.request.transfer_id, None)

        return TransferResult(
            transfer_id=state.request.transfer_id,
            success=True,
            status=TransferStatus.COMPLETED,
            transfer_type=TransferType.WARM,
            target_call_id=state.consultation_call_id,
            connected_at=state.connected_at,
            completed_at=state.completed_at,
            ring_duration_seconds=ring_duration,
            consultation_duration_seconds=consultation_duration,
            total_duration_seconds=total_duration,
        )

    async def cancel(self, state: TransferState) -> TransferResult:
        """Cancel warm transfer."""
        state.status = TransferStatus.CANCELLED
        state.completed_at = datetime.utcnow()
        state.original_on_hold = False

        self._active_transfers.pop(state.request.transfer_id, None)

        return TransferResult(
            transfer_id=state.request.transfer_id,
            success=False,
            status=TransferStatus.CANCELLED,
            transfer_type=TransferType.WARM,
            completed_at=state.completed_at,
        )

    async def introduce(self, state: TransferState) -> bool:
        """
        Start introduction (3-way call).

        Brings all parties together briefly for handoff.
        """
        if state.status != TransferStatus.CONNECTED:
            return False

        # Create conference for introduction
        state.conference_id = f"intro-{uuid.uuid4()}"
        state.original_on_hold = False

        logger.info(f"Starting introduction for warm transfer")

        return True


class ConferenceTransfer(TransferHandler):
    """
    Conference transfer.

    Adds a party to the call without disconnecting anyone.
    """

    def __init__(self):
        self._active_transfers: Dict[str, TransferState] = {}

    async def initiate(self, request: TransferRequest) -> TransferState:
        """Initiate conference (add party)."""
        state = TransferState(
            request=request,
            status=TransferStatus.INITIATED,
            original_call_id=request.call_id,
        )

        self._active_transfers[request.transfer_id] = state

        logger.info(
            f"Conference add initiated: {request.call_id} + {request.target}"
        )

        try:
            # Create or join conference
            state.conference_id = f"conf-{uuid.uuid4()}"
            state.status = TransferStatus.RINGING
            state.ring_started_at = datetime.utcnow()

            # Dial new party
            await asyncio.sleep(0.1)

            state.status = TransferStatus.CONNECTED
            state.connected_at = datetime.utcnow()
            state.transfer_party_connected = True

            return state

        except Exception as e:
            state.status = TransferStatus.FAILED
            logger.error(f"Conference add failed: {e}")
            return state

    async def complete(self, state: TransferState) -> TransferResult:
        """Complete conference add."""
        if state.status != TransferStatus.CONNECTED:
            return TransferResult(
                transfer_id=state.request.transfer_id,
                success=False,
                status=state.status,
                transfer_type=TransferType.CONFERENCE,
                error="Party not connected",
            )

        state.status = TransferStatus.COMPLETED
        state.completed_at = datetime.utcnow()

        total_duration = (state.completed_at - state.initiated_at).total_seconds()

        self._active_transfers.pop(state.request.transfer_id, None)

        return TransferResult(
            transfer_id=state.request.transfer_id,
            success=True,
            status=TransferStatus.COMPLETED,
            transfer_type=TransferType.CONFERENCE,
            connected_at=state.connected_at,
            completed_at=state.completed_at,
            total_duration_seconds=total_duration,
        )

    async def cancel(self, state: TransferState) -> TransferResult:
        """Cancel conference add (drop new party)."""
        state.status = TransferStatus.CANCELLED
        state.completed_at = datetime.utcnow()

        self._active_transfers.pop(state.request.transfer_id, None)

        return TransferResult(
            transfer_id=state.request.transfer_id,
            success=False,
            status=TransferStatus.CANCELLED,
            transfer_type=TransferType.CONFERENCE,
            completed_at=state.completed_at,
        )


class TransferManager:
    """
    Manages call transfers.
    """

    def __init__(self):
        self._handlers: Dict[TransferType, TransferHandler] = {
            TransferType.BLIND: BlindTransfer(),
            TransferType.ATTENDED: AttendedTransfer(),
            TransferType.WARM: WarmTransfer(),
            TransferType.CONFERENCE: ConferenceTransfer(),
        }

        self._active_transfers: Dict[str, TransferState] = {}
        self._lock = asyncio.Lock()

        # Callbacks
        self._on_transfer_complete: List[Callable[[TransferResult], Awaitable[None]]] = []

        # Statistics
        self._total_transfers = 0
        self._successful_transfers = 0

    async def transfer(self, request: TransferRequest) -> TransferResult:
        """Execute transfer."""
        handler = self._handlers.get(request.transfer_type)
        if not handler:
            return TransferResult(
                transfer_id=request.transfer_id,
                success=False,
                status=TransferStatus.FAILED,
                transfer_type=request.transfer_type,
                error=f"Unknown transfer type: {request.transfer_type}",
            )

        self._total_transfers += 1

        try:
            # Initiate transfer
            state = await handler.initiate(request)

            async with self._lock:
                self._active_transfers[request.transfer_id] = state

            # Wait for connection or timeout
            timeout = request.timeout_seconds
            start = datetime.utcnow()

            while state.status == TransferStatus.RINGING:
                if (datetime.utcnow() - start).total_seconds() > timeout:
                    state.status = TransferStatus.TIMEOUT
                    break
                await asyncio.sleep(0.1)

            if state.status == TransferStatus.TIMEOUT:
                return TransferResult(
                    transfer_id=request.transfer_id,
                    success=False,
                    status=TransferStatus.TIMEOUT,
                    transfer_type=request.transfer_type,
                    error="Transfer timed out",
                )

            # For blind transfer, complete immediately
            if request.transfer_type == TransferType.BLIND:
                result = await handler.complete(state)
            else:
                # For other types, return state for manual completion
                result = TransferResult(
                    transfer_id=request.transfer_id,
                    success=True,
                    status=state.status,
                    transfer_type=request.transfer_type,
                    connected_at=state.connected_at,
                )

            if result.success:
                self._successful_transfers += 1

            # Trigger callbacks
            for callback in self._on_transfer_complete:
                try:
                    await callback(result)
                except Exception as e:
                    logger.error(f"Transfer callback error: {e}")

            return result

        except Exception as e:
            logger.error(f"Transfer error: {e}")
            return TransferResult(
                transfer_id=request.transfer_id,
                success=False,
                status=TransferStatus.FAILED,
                transfer_type=request.transfer_type,
                error=str(e),
            )

    async def complete_transfer(self, transfer_id: str) -> Optional[TransferResult]:
        """Complete an active transfer."""
        async with self._lock:
            state = self._active_transfers.get(transfer_id)
            if not state:
                return None

            handler = self._handlers.get(state.request.transfer_type)
            if not handler:
                return None

            result = await handler.complete(state)
            self._active_transfers.pop(transfer_id, None)

            return result

    async def cancel_transfer(self, transfer_id: str) -> Optional[TransferResult]:
        """Cancel an active transfer."""
        async with self._lock:
            state = self._active_transfers.get(transfer_id)
            if not state:
                return None

            handler = self._handlers.get(state.request.transfer_type)
            if not handler:
                return None

            result = await handler.cancel(state)
            self._active_transfers.pop(transfer_id, None)

            return result

    async def get_transfer_state(self, transfer_id: str) -> Optional[TransferState]:
        """Get active transfer state."""
        return self._active_transfers.get(transfer_id)

    def on_transfer_complete(
        self,
        callback: Callable[[TransferResult], Awaitable[None]],
    ) -> None:
        """Register transfer complete callback."""
        self._on_transfer_complete.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get transfer statistics."""
        return {
            "total_transfers": self._total_transfers,
            "successful_transfers": self._successful_transfers,
            "success_rate": self._successful_transfers / max(1, self._total_transfers),
            "active_transfers": len(self._active_transfers),
        }

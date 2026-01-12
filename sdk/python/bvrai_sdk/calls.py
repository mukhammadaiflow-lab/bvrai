"""Calls API for Builder Engine."""

from typing import Optional, Dict, List, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import asyncio

if TYPE_CHECKING:
    from bvrai_sdk.client import BvraiClient


class CallStatus(str, Enum):
    """Call status values."""
    PENDING = "pending"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BUSY = "busy"
    NO_ANSWER = "no_answer"
    CANCELED = "canceled"


class CallDirection(str, Enum):
    """Call direction."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


@dataclass
class CallTranscript:
    """Transcript of a call turn."""
    speaker: str  # "user" or "assistant"
    text: str
    timestamp: str
    duration_ms: Optional[int] = None
    confidence: Optional[float] = None


@dataclass
class Call:
    """A phone call."""
    id: str
    agent_id: str
    direction: str
    status: str
    to_number: Optional[str] = None
    from_number: Optional[str] = None
    started_at: Optional[str] = None
    answered_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    end_reason: Optional[str] = None
    recording_url: Optional[str] = None
    transcript: List[CallTranscript] = field(default_factory=list)
    summary: Optional[str] = None
    sentiment: Optional[str] = None
    cost_cents: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Call":
        """Create Call from API response."""
        transcript = []
        for t in data.get("transcript", []):
            if isinstance(t, dict):
                transcript.append(CallTranscript(
                    speaker=t.get("speaker", "unknown"),
                    text=t.get("text", ""),
                    timestamp=t.get("timestamp", ""),
                    duration_ms=t.get("duration_ms"),
                    confidence=t.get("confidence"),
                ))

        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            direction=data.get("direction", "outbound"),
            status=data.get("status", "pending"),
            to_number=data.get("to_number") or data.get("callee_number"),
            from_number=data.get("from_number") or data.get("caller_number"),
            started_at=data.get("started_at"),
            answered_at=data.get("answered_at"),
            ended_at=data.get("ended_at"),
            duration_seconds=data.get("duration_seconds"),
            end_reason=data.get("end_reason"),
            recording_url=data.get("recording_url"),
            transcript=transcript,
            summary=data.get("summary"),
            sentiment=data.get("sentiment"),
            cost_cents=data.get("cost_cents", 0),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
        )

    @property
    def is_active(self) -> bool:
        """Check if call is still active."""
        return self.status in (
            CallStatus.PENDING.value,
            CallStatus.RINGING.value,
            CallStatus.IN_PROGRESS.value,
        )

    @property
    def is_completed(self) -> bool:
        """Check if call completed successfully."""
        return self.status == CallStatus.COMPLETED.value

    @property
    def is_failed(self) -> bool:
        """Check if call failed."""
        return self.status in (
            CallStatus.FAILED.value,
            CallStatus.BUSY.value,
            CallStatus.NO_ANSWER.value,
        )


class CallsAPI:
    """
    Calls API client.

    Make and manage phone calls.
    """

    def __init__(self, client: "BvraiClient"):
        self._client = client

    async def create(
        self,
        agent_id: str,
        to_number: str,
        from_number: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        variables: Optional[Dict[str, Any]] = None,
        max_duration_seconds: int = 1800,
        record: bool = True,
    ) -> Call:
        """
        Create an outbound call.

        Args:
            agent_id: Agent to use for the call
            to_number: Phone number to call (E.164 format)
            from_number: Caller ID (must be verified)
            metadata: Custom metadata
            variables: Variables to inject into prompts
            max_duration_seconds: Maximum call duration
            record: Whether to record the call

        Returns:
            Created Call

        Example:
            call = await client.calls.create(
                agent_id="agent-123",
                to_number="+15551234567",
                variables={"customer_name": "John"},
            )
        """
        data = {
            "agent_id": agent_id,
            "to_number": to_number,
            "from_number": from_number,
            "metadata": metadata or {},
            "variables": variables or {},
            "max_duration_seconds": max_duration_seconds,
            "record": record,
        }

        response = await self._client.post("/v1/calls", data=data)
        return Call.from_dict(response)

    async def get(self, call_id: str) -> Call:
        """
        Get a call by ID.

        Args:
            call_id: Call ID

        Returns:
            Call

        Raises:
            NotFoundError: If call not found
        """
        response = await self._client.get(f"/v1/calls/{call_id}")
        return Call.from_dict(response)

    async def list(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        direction: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Call]:
        """
        List calls with optional filters.

        Args:
            agent_id: Filter by agent
            status: Filter by status
            direction: Filter by direction
            from_date: Filter by start date (ISO format)
            to_date: Filter by end date (ISO format)
            limit: Maximum calls to return
            offset: Pagination offset

        Returns:
            List of Calls
        """
        params = {"limit": limit, "offset": offset}
        if agent_id:
            params["agent_id"] = agent_id
        if status:
            params["status"] = status
        if direction:
            params["direction"] = direction
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        response = await self._client.get("/v1/calls", params=params)
        return [Call.from_dict(c) for c in response.get("calls", [])]

    async def cancel(self, call_id: str) -> Call:
        """
        Cancel a pending or active call.

        Args:
            call_id: Call ID to cancel

        Returns:
            Updated Call

        Raises:
            ValidationError: If call cannot be canceled
        """
        response = await self._client.post(f"/v1/calls/{call_id}/cancel")
        return Call.from_dict(response)

    async def transfer(
        self,
        call_id: str,
        to_number: str,
        announce: Optional[str] = None,
    ) -> Call:
        """
        Transfer an active call.

        Args:
            call_id: Call ID to transfer
            to_number: Number to transfer to
            announce: Message to announce before transfer

        Returns:
            Updated Call
        """
        data = {
            "to_number": to_number,
            "announce": announce,
        }
        response = await self._client.post(f"/v1/calls/{call_id}/transfer", data=data)
        return Call.from_dict(response)

    async def send_dtmf(self, call_id: str, digits: str) -> bool:
        """
        Send DTMF tones to an active call.

        Args:
            call_id: Call ID
            digits: DTMF digits to send

        Returns:
            True if sent successfully
        """
        await self._client.post(
            f"/v1/calls/{call_id}/dtmf",
            data={"digits": digits},
        )
        return True

    async def get_recording(self, call_id: str) -> Optional[str]:
        """
        Get recording URL for a completed call.

        Args:
            call_id: Call ID

        Returns:
            Recording URL or None
        """
        call = await self.get(call_id)
        return call.recording_url

    async def get_transcript(self, call_id: str) -> List[CallTranscript]:
        """
        Get full transcript for a call.

        Args:
            call_id: Call ID

        Returns:
            List of transcript turns
        """
        response = await self._client.get(f"/v1/calls/{call_id}/transcript")
        return [
            CallTranscript(
                speaker=t["speaker"],
                text=t["text"],
                timestamp=t["timestamp"],
                duration_ms=t.get("duration_ms"),
                confidence=t.get("confidence"),
            )
            for t in response.get("transcript", [])
        ]

    async def wait_for_completion(
        self,
        call_id: str,
        timeout_seconds: int = 1800,
        poll_interval: float = 2.0,
    ) -> Call:
        """
        Wait for a call to complete.

        Args:
            call_id: Call ID to wait for
            timeout_seconds: Maximum wait time
            poll_interval: Polling interval in seconds

        Returns:
            Completed Call

        Raises:
            TimeoutError: If call doesn't complete in time
        """
        import time

        start_time = time.time()

        while True:
            call = await self.get(call_id)

            if not call.is_active:
                return call

            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                raise TimeoutError(f"Call {call_id} did not complete within {timeout_seconds}s")

            await asyncio.sleep(poll_interval)

    async def batch_create(
        self,
        agent_id: str,
        phone_numbers: List[str],
        from_number: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        delay_seconds: float = 1.0,
    ) -> List[Call]:
        """
        Create multiple outbound calls.

        Args:
            agent_id: Agent to use
            phone_numbers: List of numbers to call
            from_number: Caller ID
            variables: Variables for all calls
            delay_seconds: Delay between calls

        Returns:
            List of created Calls
        """
        calls = []

        for number in phone_numbers:
            call = await self.create(
                agent_id=agent_id,
                to_number=number,
                from_number=from_number,
                variables=variables,
            )
            calls.append(call)

            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

        return calls

    async def get_analytics(
        self,
        agent_id: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get call analytics.

        Args:
            agent_id: Filter by agent
            from_date: Start date
            to_date: End date

        Returns:
            Analytics data
        """
        params = {}
        if agent_id:
            params["agent_id"] = agent_id
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        return await self._client.get("/v1/calls/analytics", params=params)

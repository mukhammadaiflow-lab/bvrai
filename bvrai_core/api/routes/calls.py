"""
Call API Routes

This module provides REST API endpoints for managing voice calls.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field

from ..base import (
    APIResponse,
    ListResponse,
    NotFoundError,
    ValidationError,
    success_response,
    paginated_response,
)
from ..auth import (
    AuthContext,
    Permission,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calls", tags=["Calls"])


# =============================================================================
# Request/Response Models
# =============================================================================


class CallDirection(str):
    """Call direction types."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


class CallStatus(str):
    """Call status types."""

    QUEUED = "queued"
    RINGING = "ringing"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BUSY = "busy"
    NO_ANSWER = "no-answer"
    CANCELED = "canceled"


class OutboundCallRequest(BaseModel):
    """Request to initiate an outbound call."""

    agent_id: str = Field(..., description="Agent ID to handle the call")
    to_phone_number: str = Field(
        ...,
        description="Phone number to call (E.164 format)",
    )
    from_phone_number: str = Field(
        ...,
        description="Caller ID phone number (E.164 format)",
    )

    # Optional configuration
    first_message: Optional[str] = Field(
        default=None,
        description="Custom first message (overrides agent default)",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context variables for the call",
    )

    # Call settings
    max_duration_seconds: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Maximum call duration",
    )
    record: bool = Field(
        default=True,
        description="Record the call",
    )
    amd_enabled: bool = Field(
        default=True,
        description="Enable answering machine detection",
    )
    amd_action: str = Field(
        default="leave_message",
        description="Action when voicemail detected",
    )

    # Scheduling
    schedule_time: Optional[datetime] = Field(
        default=None,
        description="Schedule call for future time",
    )

    # Webhook
    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for call events",
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata",
    )


class CallResponse(BaseModel):
    """Call response model."""

    id: str
    organization_id: str
    agent_id: str

    # Direction and status
    direction: str
    status: str

    # Phone numbers
    from_phone_number: str
    to_phone_number: str

    # Timing
    started_at: Optional[datetime] = None
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None

    # Recording
    recording_url: Optional[str] = None
    recording_duration_seconds: Optional[int] = None

    # Transcript
    transcript_id: Optional[str] = None
    transcript_available: bool = False

    # Cost
    cost_cents: Optional[int] = None

    # Call quality
    amd_result: Optional[str] = None  # human, voicemail, unknown

    # Context
    context: Dict[str, Any] = {}

    # Metadata
    metadata: Dict[str, Any] = {}

    # Timestamps
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class CallSummary(BaseModel):
    """Summary of a call for list views."""

    id: str
    agent_id: str
    direction: str
    status: str
    from_phone_number: str
    to_phone_number: str
    duration_seconds: Optional[int] = None
    created_at: datetime


class TranscriptSegment(BaseModel):
    """A segment of call transcript."""

    speaker: str  # "agent" or "customer"
    text: str
    start_time_ms: int
    end_time_ms: int
    confidence: float = 1.0


class TranscriptResponse(BaseModel):
    """Call transcript response."""

    call_id: str
    segments: List[TranscriptSegment]
    full_text: str
    duration_seconds: int
    language: str = "en"
    created_at: datetime


class CallEndRequest(BaseModel):
    """Request to end an active call."""

    reason: Optional[str] = Field(
        default=None,
        description="Reason for ending the call",
    )


class CallTransferRequest(BaseModel):
    """Request to transfer an active call."""

    to_phone_number: str = Field(
        ...,
        description="Phone number to transfer to",
    )
    announce: bool = Field(
        default=True,
        description="Announce transfer to customer",
    )
    announcement_message: Optional[str] = Field(
        default=None,
        description="Custom announcement message",
    )


class CallMessageRequest(BaseModel):
    """Request to send a message during a call."""

    message: str = Field(
        ...,
        description="Message to speak",
    )
    interrupt: bool = Field(
        default=False,
        description="Interrupt current speech",
    )


# =============================================================================
# Routes
# =============================================================================


@router.post(
    "/outbound",
    response_model=APIResponse[CallResponse],
    status_code=201,
    summary="Create Outbound Call",
    description="Initiate an outbound call using an agent.",
)
async def create_outbound_call(
    request: OutboundCallRequest,
    auth: AuthContext = Depends(),
):
    """Create an outbound call."""
    auth.require_permission(Permission.CALLS_WRITE)

    # In production, this would:
    # 1. Validate agent exists and is active
    # 2. Validate phone numbers
    # 3. Check credits/balance
    # 4. Queue or initiate call via telephony provider

    call = CallResponse(
        id="call_" + "x" * 24,
        organization_id=auth.organization_id,
        agent_id=request.agent_id,
        direction=CallDirection.OUTBOUND,
        status=CallStatus.QUEUED,
        from_phone_number=request.from_phone_number,
        to_phone_number=request.to_phone_number,
        context=request.context,
        metadata=request.metadata,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    return success_response(call.dict())


@router.get(
    "",
    response_model=ListResponse[CallSummary],
    summary="List Calls",
    description="List all calls for the organization.",
)
async def list_calls(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    agent_id: Optional[str] = Query(None, description="Filter by agent"),
    direction: Optional[str] = Query(None, description="Filter by direction"),
    status: Optional[str] = Query(None, description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    phone_number: Optional[str] = Query(None, description="Filter by phone number"),
    auth: AuthContext = Depends(),
):
    """List all calls."""
    auth.require_permission(Permission.CALLS_READ)

    # In production, this would query the database with filters
    calls = []

    return paginated_response(
        items=[c.dict() for c in calls],
        page=page,
        page_size=page_size,
        total_items=0,
    )


@router.get(
    "/{call_id}",
    response_model=APIResponse[CallResponse],
    summary="Get Call",
    description="Get details of a specific call.",
)
async def get_call(
    call_id: str = Path(..., description="Call ID"),
    auth: AuthContext = Depends(),
):
    """Get call by ID."""
    auth.require_permission(Permission.CALLS_READ)

    raise NotFoundError("Call", call_id)


@router.get(
    "/{call_id}/transcript",
    response_model=APIResponse[TranscriptResponse],
    summary="Get Call Transcript",
    description="Get the transcript of a call.",
)
async def get_call_transcript(
    call_id: str = Path(..., description="Call ID"),
    auth: AuthContext = Depends(),
):
    """Get call transcript."""
    auth.require_permission(Permission.TRANSCRIPTS_READ)

    raise NotFoundError("Call", call_id)


@router.get(
    "/{call_id}/recording",
    response_model=APIResponse[Dict[str, Any]],
    summary="Get Call Recording",
    description="Get the recording URL for a call.",
)
async def get_call_recording(
    call_id: str = Path(..., description="Call ID"),
    auth: AuthContext = Depends(),
):
    """Get call recording."""
    auth.require_permission(Permission.RECORDINGS_READ)

    raise NotFoundError("Call", call_id)


@router.delete(
    "/{call_id}/recording",
    status_code=204,
    summary="Delete Call Recording",
    description="Delete a call recording.",
)
async def delete_call_recording(
    call_id: str = Path(..., description="Call ID"),
    auth: AuthContext = Depends(),
):
    """Delete call recording."""
    auth.require_permission(Permission.RECORDINGS_DELETE)

    raise NotFoundError("Call", call_id)


@router.post(
    "/{call_id}/end",
    response_model=APIResponse[CallResponse],
    summary="End Call",
    description="End an active call.",
)
async def end_call(
    call_id: str = Path(..., description="Call ID"),
    request: CallEndRequest = Body(default=CallEndRequest()),
    auth: AuthContext = Depends(),
):
    """End an active call."""
    auth.require_permission(Permission.CALLS_WRITE)

    # In production, this would:
    # 1. Check call is active
    # 2. Send hangup command to telephony provider
    # 3. Update call status

    raise NotFoundError("Call", call_id)


@router.post(
    "/{call_id}/transfer",
    response_model=APIResponse[CallResponse],
    summary="Transfer Call",
    description="Transfer an active call to another phone number.",
)
async def transfer_call(
    call_id: str = Path(..., description="Call ID"),
    request: CallTransferRequest = Body(...),
    auth: AuthContext = Depends(),
):
    """Transfer an active call."""
    auth.require_permission(Permission.CALLS_WRITE)

    raise NotFoundError("Call", call_id)


@router.post(
    "/{call_id}/message",
    response_model=APIResponse[Dict[str, Any]],
    summary="Send Message",
    description="Send a message to be spoken during an active call.",
)
async def send_call_message(
    call_id: str = Path(..., description="Call ID"),
    request: CallMessageRequest = Body(...),
    auth: AuthContext = Depends(),
):
    """Send a message during a call."""
    auth.require_permission(Permission.CALLS_WRITE)

    raise NotFoundError("Call", call_id)


@router.post(
    "/{call_id}/mute",
    response_model=APIResponse[Dict[str, Any]],
    summary="Mute Agent",
    description="Mute the agent on an active call.",
)
async def mute_agent(
    call_id: str = Path(..., description="Call ID"),
    auth: AuthContext = Depends(),
):
    """Mute agent on call."""
    auth.require_permission(Permission.CALLS_WRITE)

    raise NotFoundError("Call", call_id)


@router.post(
    "/{call_id}/unmute",
    response_model=APIResponse[Dict[str, Any]],
    summary="Unmute Agent",
    description="Unmute the agent on an active call.",
)
async def unmute_agent(
    call_id: str = Path(..., description="Call ID"),
    auth: AuthContext = Depends(),
):
    """Unmute agent on call."""
    auth.require_permission(Permission.CALLS_WRITE)

    raise NotFoundError("Call", call_id)


@router.get(
    "/{call_id}/events",
    response_model=ListResponse[Dict[str, Any]],
    summary="Get Call Events",
    description="Get events/timeline for a call.",
)
async def get_call_events(
    call_id: str = Path(..., description="Call ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    auth: AuthContext = Depends(),
):
    """Get call events/timeline."""
    auth.require_permission(Permission.CALLS_READ)

    raise NotFoundError("Call", call_id)


@router.get(
    "/active",
    response_model=ListResponse[CallSummary],
    summary="List Active Calls",
    description="List all currently active calls.",
)
async def list_active_calls(
    agent_id: Optional[str] = Query(None, description="Filter by agent"),
    auth: AuthContext = Depends(),
):
    """List active calls."""
    auth.require_permission(Permission.CALLS_READ)

    return paginated_response(
        items=[],
        page=1,
        page_size=100,
        total_items=0,
    )


@router.post(
    "/batch",
    response_model=APIResponse[Dict[str, Any]],
    status_code=201,
    summary="Create Batch Calls",
    description="Create multiple outbound calls in batch.",
)
async def create_batch_calls(
    calls: List[OutboundCallRequest] = Body(...),
    auth: AuthContext = Depends(),
):
    """Create batch outbound calls."""
    auth.require_permission(Permission.CALLS_WRITE)
    auth.require_permission(Permission.CAMPAIGNS_EXECUTE)

    if len(calls) > 100:
        raise ValidationError("Maximum 100 calls per batch")

    # In production, this would queue all calls for processing

    return success_response({
        "queued": len(calls),
        "batch_id": "batch_" + "x" * 24,
    })

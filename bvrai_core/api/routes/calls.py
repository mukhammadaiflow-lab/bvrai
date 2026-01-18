"""
Call API Routes

This module provides REST API endpoints for managing voice calls.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

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
from ..dependencies import get_db_session, get_auth_context
from ...database.repositories import CallRepository, AgentRepository


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calls", tags=["Calls"])


# =============================================================================
# Enums
# =============================================================================


class CallDirection(str, Enum):
    """Call direction types."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class CallStatus(str, Enum):
    """Call status types."""
    QUEUED = "queued"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BUSY = "busy"
    NO_ANSWER = "no_answer"
    CANCELED = "canceled"


# =============================================================================
# Request/Response Models
# =============================================================================


class OutboundCallRequest(BaseModel):
    """Request to initiate an outbound call."""

    agent_id: str = Field(..., description="Agent ID to handle the call")
    to_phone_number: str = Field(
        ...,
        description="Phone number to call (E.164 format)",
    )
    from_phone_number: Optional[str] = Field(
        default=None,
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
    from_phone_number: Optional[str] = None
    to_phone_number: str

    # Duration
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None

    # Recording
    recording_url: Optional[str] = None
    transcript_url: Optional[str] = None

    # Cost
    cost_cents: Optional[int] = None

    # Context
    context: Dict[str, Any] = {}

    # Metadata
    metadata: Dict[str, Any] = {}

    # Timestamps
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CallSummary(BaseModel):
    """Summary of a call for list views."""

    id: str
    agent_id: str
    direction: str
    status: str
    from_phone_number: Optional[str] = None
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


# =============================================================================
# Helper Functions
# =============================================================================


def call_to_response(call) -> dict:
    """Convert database call model to response dict."""
    return {
        "id": call.id,
        "organization_id": call.organization_id,
        "agent_id": call.agent_id,
        "direction": call.direction,
        "status": call.status,
        "from_phone_number": call.from_number,
        "to_phone_number": call.to_number,
        "started_at": call.initiated_at,
        "ended_at": call.ended_at,
        "duration_seconds": int(call.duration_seconds) if call.duration_seconds else None,
        "recording_url": call.recording_url,
        "transcript_url": call.transcript_url,
        "cost_cents": int(call.cost_amount * 100) if call.cost_amount else None,
        "context": {},
        "metadata": call.metadata_json or {},
        "created_at": call.created_at,
        "updated_at": call.updated_at,
    }


def call_to_summary(call) -> dict:
    """Convert database call model to summary dict."""
    return {
        "id": call.id,
        "agent_id": call.agent_id,
        "direction": call.direction,
        "status": call.status,
        "from_phone_number": call.from_number,
        "to_phone_number": call.to_number,
        "duration_seconds": int(call.duration_seconds) if call.duration_seconds else None,
        "created_at": call.created_at,
    }


# =============================================================================
# Routes
# =============================================================================


@router.post(
    "/outbound",
    response_model=APIResponse[CallResponse],
    status_code=201,
    summary="Initiate Outbound Call",
    description="Start an outbound call using an agent.",
)
async def create_outbound_call(
    request: OutboundCallRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Initiate an outbound call."""
    auth.require_permission(Permission.CALLS_WRITE)

    # Verify agent exists and belongs to organization
    agent_repo = AgentRepository(db)
    agent = await agent_repo.get_by_id(request.agent_id)

    if not agent or agent.organization_id != auth.organization_id:
        raise NotFoundError("Agent", request.agent_id)

    # Create call record
    call_repo = CallRepository(db)

    # Determine from number - use request value, agent's number, or default
    from_number = request.from_phone_number
    if not from_number:
        from_number = getattr(agent, 'phone_number', None)
    if not from_number:
        # Use a default/placeholder number for MVP
        from_number = "+10000000000"

    call = await call_repo.create(
        organization_id=auth.organization_id,
        agent_id=request.agent_id,
        direction=CallDirection.OUTBOUND.value,
        status=CallStatus.QUEUED.value,
        from_number=from_number,
        to_number=request.to_phone_number,
        metadata_json=request.metadata,
    )

    logger.info(f"Created outbound call {call.id} to {request.to_phone_number}")

    # TODO: In production, initiate actual call via Twilio/telephony provider
    # For now, we just create the record

    return success_response(call_to_response(call))


# Alias: POST /calls for frontend compatibility
@router.post(
    "",
    response_model=APIResponse[CallResponse],
    status_code=201,
    summary="Initiate Call",
    description="Initiate an outbound call (alias for /outbound).",
    include_in_schema=False,  # Hide from docs to avoid duplication
)
async def initiate_call_alias(
    request: OutboundCallRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Alias for create_outbound_call."""
    return await create_outbound_call(request, auth, db)


@router.get(
    "",
    response_model=ListResponse[CallSummary],
    summary="List Calls",
    description="List all calls for the organization.",
)
async def list_calls(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    agent_id: Optional[str] = Query(None),
    direction: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """List all calls."""
    auth.require_permission(Permission.CALLS_READ)

    repo = CallRepository(db)

    skip = (page - 1) * page_size
    calls = await repo.list_by_organization(
        organization_id=auth.organization_id,
        agent_id=agent_id,
        status=status,
        start_date=from_date,
        end_date=to_date,
        skip=skip,
        limit=page_size,
    )

    # Get total count
    total = await repo.count_by_organization(
        organization_id=auth.organization_id,
        agent_id=agent_id,
        status=status,
    )

    return paginated_response(
        items=[call_to_summary(c) for c in calls],
        page=page,
        page_size=page_size,
        total_items=total,
    )


@router.get(
    "/{call_id}",
    response_model=APIResponse[CallResponse],
    summary="Get Call",
    description="Get details of a specific call.",
)
async def get_call(
    call_id: str = Path(..., description="Call ID"),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get call by ID."""
    auth.require_permission(Permission.CALLS_READ)

    repo = CallRepository(db)
    call = await repo.get_by_id(call_id)

    if not call:
        raise NotFoundError("Call", call_id)

    if call.organization_id != auth.organization_id:
        raise NotFoundError("Call", call_id)

    return success_response(call_to_response(call))


@router.get(
    "/{call_id}/transcript",
    response_model=APIResponse[TranscriptResponse],
    summary="Get Call Transcript",
    description="Get the transcript of a call.",
)
async def get_call_transcript(
    call_id: str = Path(..., description="Call ID"),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get call transcript."""
    auth.require_permission(Permission.CALLS_READ)

    repo = CallRepository(db)
    call = await repo.get_by_id(call_id)

    if not call:
        raise NotFoundError("Call", call_id)

    if call.organization_id != auth.organization_id:
        raise NotFoundError("Call", call_id)

    # Get transcript from events
    events = await repo.get_events(call_id, event_type="transcript")

    segments = []
    full_text_parts = []

    for event in events:
        event_data = event.event_data or {}
        segment = TranscriptSegment(
            speaker=event_data.get("speaker", "unknown"),
            text=event_data.get("text", ""),
            start_time_ms=event_data.get("start_time_ms", 0),
            end_time_ms=event_data.get("end_time_ms", 0),
            confidence=event_data.get("confidence", 1.0),
        )
        segments.append(segment)
        full_text_parts.append(f"{segment.speaker}: {segment.text}")

    transcript = TranscriptResponse(
        call_id=call_id,
        segments=segments,
        full_text="\n".join(full_text_parts),
        duration_seconds=call.duration_seconds or 0,
        language="en",
        created_at=call.created_at,
    )

    return success_response(transcript.dict())


@router.post(
    "/{call_id}/end",
    response_model=APIResponse[CallResponse],
    summary="End Call",
    description="End an active call.",
)
async def end_call(
    call_id: str = Path(..., description="Call ID"),
    request: CallEndRequest = Body(default=CallEndRequest()),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """End an active call."""
    auth.require_permission(Permission.CALLS_WRITE)

    repo = CallRepository(db)
    call = await repo.get_by_id(call_id)

    if not call:
        raise NotFoundError("Call", call_id)

    if call.organization_id != auth.organization_id:
        raise NotFoundError("Call", call_id)

    # Update call status
    call = await repo.update(
        call_id,
        status=CallStatus.COMPLETED.value,
        ended_at=datetime.utcnow(),
    )

    # Calculate duration
    if call.initiated_at and call.ended_at:
        duration = int((call.ended_at - call.initiated_at).total_seconds())
        call = await repo.update(call_id, duration_seconds=duration)

    # Add end event
    await repo.add_event(
        call_id=call_id,
        event_type="call_ended",
        event_data={"reason": request.reason},
    )

    logger.info(f"Ended call {call_id}")

    # TODO: In production, hang up actual call via telephony provider

    return success_response(call_to_response(call))


@router.post(
    "/{call_id}/transfer",
    response_model=APIResponse[CallResponse],
    summary="Transfer Call",
    description="Transfer an active call to another number.",
)
async def transfer_call(
    call_id: str = Path(..., description="Call ID"),
    request: CallTransferRequest = Body(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Transfer an active call."""
    auth.require_permission(Permission.CALLS_WRITE)

    repo = CallRepository(db)
    call = await repo.get_by_id(call_id)

    if not call:
        raise NotFoundError("Call", call_id)

    if call.organization_id != auth.organization_id:
        raise NotFoundError("Call", call_id)

    if call.status != CallStatus.IN_PROGRESS.value:
        raise ValidationError(
            "transfer_error",
            f"Cannot transfer call in status: {call.status}"
        )

    # Add transfer event
    await repo.add_event(
        call_id=call_id,
        event_type="transfer_initiated",
        event_data={
            "to_phone_number": request.to_phone_number,
            "announce": request.announce,
            "announcement_message": request.announcement_message,
        },
    )

    logger.info(f"Initiated transfer of call {call_id} to {request.to_phone_number}")

    # TODO: In production, initiate actual transfer via telephony provider

    return success_response(call_to_response(call))


@router.get(
    "/{call_id}/events",
    response_model=ListResponse[dict],
    summary="Get Call Events",
    description="Get events/timeline for a call.",
)
async def get_call_events(
    call_id: str = Path(..., description="Call ID"),
    event_type: Optional[str] = Query(None),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get call events."""
    auth.require_permission(Permission.CALLS_READ)

    repo = CallRepository(db)
    call = await repo.get_by_id(call_id)

    if not call:
        raise NotFoundError("Call", call_id)

    if call.organization_id != auth.organization_id:
        raise NotFoundError("Call", call_id)

    events = await repo.get_events(call_id, event_type=event_type)

    event_dicts = [
        {
            "id": e.id,
            "call_id": e.call_id,
            "event_type": e.event_type,
            "event_data": e.event_data,
            "timestamp": e.timestamp,
        }
        for e in events
    ]

    return paginated_response(
        items=event_dicts,
        page=1,
        page_size=len(event_dicts),
        total_items=len(event_dicts),
    )

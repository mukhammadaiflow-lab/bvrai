"""Call schemas for API requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.database.models import CallStatus, CallDirection


class CallCreate(BaseModel):
    """Schema for creating a new outbound call."""

    agent_id: UUID
    to_number: str = Field(..., description="Phone number to call (E.164 format)")
    from_number: Optional[str] = Field(None, description="Caller ID (defaults to agent's number)")
    metadata: Optional[dict] = Field(default_factory=dict)


class CallResponse(BaseModel):
    """Schema for call response."""

    id: UUID
    agent_id: UUID
    session_id: str
    twilio_call_sid: Optional[str] = None
    direction: CallDirection
    status: CallStatus
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CallListResponse(BaseModel):
    """Paginated list of calls."""

    calls: list[CallResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class CallLogEntry(BaseModel):
    """Schema for call log entry."""

    id: UUID
    call_id: UUID
    timestamp: datetime
    event_type: str
    speaker: Optional[str] = None
    content: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class CallTranscript(BaseModel):
    """Full call transcript."""

    call_id: UUID
    entries: list[CallLogEntry]
    duration_seconds: Optional[int] = None


class CallSummary(BaseModel):
    """AI-generated call summary."""

    call_id: UUID
    summary: str
    key_points: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    sentiment: Optional[str] = None
    topics: list[str] = Field(default_factory=list)


class CallWebhookEvent(BaseModel):
    """Webhook event for call status updates."""

    event_type: str
    call_id: UUID
    session_id: str
    timestamp: datetime
    data: dict = Field(default_factory=dict)


class OutboundCallRequest(BaseModel):
    """Request to initiate an outbound call."""

    agent_id: UUID
    to_number: str
    from_number: Optional[str] = None
    webhook_url: Optional[str] = None
    metadata: Optional[dict] = None


class OutboundCallResponse(BaseModel):
    """Response after initiating outbound call."""

    call_id: UUID
    session_id: str
    status: CallStatus
    message: str


class TransferCallRequest(BaseModel):
    """Request to transfer a call."""

    target_number: str = Field(..., description="Number to transfer to")
    announce: bool = Field(default=True, description="Announce transfer to caller")
    message: Optional[str] = Field(None, description="Message before transfer")


class RecordingResponse(BaseModel):
    """Call recording information."""

    call_id: UUID
    recording_url: Optional[str] = None
    recording_duration: Optional[int] = None
    recording_status: str = "not_available"

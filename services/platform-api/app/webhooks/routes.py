"""Webhook routes for external service callbacks."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Header, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.database.models import CallStatus, CallDirection
from app.calls.service import CallService

import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# =====================
# Twilio Webhooks
# =====================


class TwilioVoiceRequest(BaseModel):
    """Twilio voice webhook request."""

    CallSid: str
    AccountSid: str
    From: str
    To: str
    CallStatus: str
    Direction: Optional[str] = None
    CallerName: Optional[str] = None
    ForwardedFrom: Optional[str] = None


class TwilioStatusCallback(BaseModel):
    """Twilio status callback request."""

    CallSid: str
    CallStatus: str
    CallDuration: Optional[str] = None
    RecordingUrl: Optional[str] = None
    RecordingSid: Optional[str] = None


@router.post("/twilio/voice")
async def twilio_voice_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Handle incoming Twilio voice call."""
    form_data = await request.form()
    data = dict(form_data)

    logger.info("Twilio voice webhook", data=data)

    call_sid = data.get("CallSid")
    from_number = data.get("From")
    to_number = data.get("To")
    call_status = data.get("CallStatus")

    # Map Twilio status to our status
    status_map = {
        "queued": CallStatus.QUEUED,
        "ringing": CallStatus.RINGING,
        "in-progress": CallStatus.IN_PROGRESS,
        "completed": CallStatus.COMPLETED,
        "failed": CallStatus.FAILED,
        "busy": CallStatus.FAILED,
        "no-answer": CallStatus.FAILED,
        "canceled": CallStatus.FAILED,
    }

    service = CallService(db)

    # Check if call already exists
    existing = await service.get_by_twilio_sid(call_sid)

    if existing:
        # Update status
        await service.update_status(
            existing.id,
            status_map.get(call_status, CallStatus.IN_PROGRESS),
        )
    else:
        # TODO: Get agent_id from phone number routing
        # For now, this needs to be handled by telephony gateway
        pass

    await db.commit()

    # Return TwiML to connect to media stream
    # This should be handled by telephony gateway
    return {
        "message": "Webhook received",
        "call_sid": call_sid,
    }


@router.post("/twilio/status")
async def twilio_status_callback(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Handle Twilio call status updates."""
    form_data = await request.form()
    data = dict(form_data)

    logger.info("Twilio status callback", data=data)

    call_sid = data.get("CallSid")
    call_status = data.get("CallStatus")
    duration = data.get("CallDuration")

    status_map = {
        "completed": CallStatus.COMPLETED,
        "failed": CallStatus.FAILED,
        "busy": CallStatus.FAILED,
        "no-answer": CallStatus.FAILED,
        "canceled": CallStatus.FAILED,
    }

    service = CallService(db)
    call = await service.get_by_twilio_sid(call_sid)

    if call:
        new_status = status_map.get(call_status)
        if new_status:
            ended_at = datetime.utcnow() if new_status in [
                CallStatus.COMPLETED,
                CallStatus.FAILED,
            ] else None

            await service.update_status(call.id, new_status, ended_at=ended_at)

            # Log recording URL if present
            recording_url = data.get("RecordingUrl")
            if recording_url:
                await service.add_log(
                    call.id,
                    event_type="recording",
                    content=recording_url,
                    metadata={
                        "recording_sid": data.get("RecordingSid"),
                        "duration": duration,
                    },
                )

    await db.commit()

    return {"status": "ok"}


# =====================
# Internal Service Webhooks
# =====================


class CallEventPayload(BaseModel):
    """Internal call event payload."""

    session_id: str
    event_type: str
    timestamp: datetime = None
    data: Optional[dict] = None


@router.post("/internal/call-event")
async def internal_call_event(
    payload: CallEventPayload,
    db: AsyncSession = Depends(get_db),
    x_service_key: str = Header(..., alias="X-Service-Key"),
):
    """Handle internal call events from other services."""
    # TODO: Validate service key

    service = CallService(db)
    call = await service.get_by_session(payload.session_id)

    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with session {payload.session_id} not found",
        )

    # Log the event
    await service.add_log(
        call.id,
        event_type=payload.event_type,
        content=str(payload.data) if payload.data else None,
        metadata=payload.data,
    )

    # Update status based on event type
    if payload.event_type == "call_started":
        await service.update_status(call.id, CallStatus.IN_PROGRESS)
    elif payload.event_type == "call_ended":
        await service.end_call(call.id)
    elif payload.event_type == "call_failed":
        await service.end_call(call.id, CallStatus.FAILED)

    await db.commit()

    return {"status": "ok", "call_id": str(call.id)}


class TranscriptPayload(BaseModel):
    """Transcript event payload."""

    session_id: str
    speaker: str  # "user" or "agent"
    text: str
    timestamp: datetime = None
    is_final: bool = True


@router.post("/internal/transcript")
async def internal_transcript(
    payload: TranscriptPayload,
    db: AsyncSession = Depends(get_db),
    x_service_key: str = Header(..., alias="X-Service-Key"),
):
    """Handle transcript events from conversation engine."""
    service = CallService(db)
    call = await service.get_by_session(payload.session_id)

    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with session {payload.session_id} not found",
        )

    event_type = "user_speech" if payload.speaker == "user" else "agent_speech"

    await service.add_log(
        call.id,
        event_type=event_type,
        speaker=payload.speaker,
        content=payload.text,
        metadata={"is_final": payload.is_final},
    )

    await db.commit()

    return {"status": "ok"}


# =====================
# Health Check
# =====================


@router.get("/health")
async def webhook_health():
    """Webhook endpoint health check."""
    return {"status": "healthy", "service": "webhooks"}

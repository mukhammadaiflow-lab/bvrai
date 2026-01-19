"""Call API routes."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.database.models import CallStatus, CallDirection
from app.database.session import get_db
from app.calls.schemas import (
    CallResponse,
    CallListResponse,
    CallLogEntry,
    CallTranscript,
    CallSummary,
    OutboundCallRequest,
    OutboundCallResponse,
    TransferCallRequest,
    RecordingResponse,
)
from app.calls.service import CallService
from app.auth.dependencies import get_current_user_id
from app.telephony import (
    get_twilio_client,
    OutboundCallParams,
    TransferParams,
    TransferType,
)
from app.config import get_settings

logger = structlog.get_logger()

router = APIRouter(prefix="/calls", tags=["calls"])


@router.get("", response_model=CallListResponse)
async def list_calls(
    agent_id: Optional[UUID] = Query(None, description="Filter by agent"),
    status: Optional[CallStatus] = Query(None, description="Filter by status"),
    direction: Optional[CallDirection] = Query(None, description="Filter by direction"),
    from_date: Optional[datetime] = Query(None, description="Filter from date"),
    to_date: Optional[datetime] = Query(None, description="Filter to date"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """List calls with optional filters."""
    service = CallService(db)
    calls, total = await service.list(
        agent_id=agent_id,
        status=status,
        direction=direction,
        from_date=from_date,
        to_date=to_date,
        page=page,
        page_size=page_size,
    )

    return CallListResponse(
        calls=[CallResponse.model_validate(c) for c in calls],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/active", response_model=list[CallResponse])
async def get_active_calls(
    agent_id: Optional[UUID] = Query(None, description="Filter by agent"),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get all currently active calls."""
    service = CallService(db)
    calls = await service.get_active_calls(agent_id=agent_id)
    return [CallResponse.model_validate(c) for c in calls]


@router.post("/outbound", response_model=OutboundCallResponse)
async def initiate_outbound_call(
    request: OutboundCallRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Initiate an outbound call."""
    service = CallService(db)
    settings = get_settings()
    twilio_client = get_twilio_client()

    # Create call record first
    call = await service.create(
        agent_id=request.agent_id,
        direction=CallDirection.OUTBOUND,
        from_number=request.from_number,
        to_number=request.to_number,
        metadata=request.metadata,
    )

    await db.flush()

    # Build outbound call parameters
    outbound_params = OutboundCallParams(
        to_number=request.to_number,
        from_number=request.from_number,
        agent_id=str(request.agent_id),
        session_id=call.session_id,
        webhook_url=request.webhook_url,
        status_callback_url=f"{settings.twilio_webhook_base_url}/api/v1/webhooks/twilio/status" if settings.twilio_webhook_base_url else None,
        record=settings.twilio_recording_enabled,
        metadata=request.metadata or {},
    )

    # Initiate call via Twilio
    result = await twilio_client.initiate_outbound_call(outbound_params)

    if result.success:
        # Update call record with Twilio SID
        call.twilio_call_sid = result.call_sid
        call.status = CallStatus.INITIATED

        await service.add_log(
            call_id=call.id,
            event_type="call_initiated",
            content=f"Outbound call initiated to {request.to_number}",
            metadata={
                "twilio_call_sid": result.call_sid,
                "from_number": request.from_number,
                "to_number": request.to_number,
            },
        )

        await db.commit()

        logger.info(
            "Outbound call initiated",
            call_id=str(call.id),
            twilio_sid=result.call_sid,
            to_number=request.to_number,
        )

        return OutboundCallResponse(
            call_id=call.id,
            session_id=call.session_id,
            status=call.status,
            message="Call initiated successfully",
        )
    else:
        # Call initiation failed
        call.status = CallStatus.FAILED

        await service.add_log(
            call_id=call.id,
            event_type="call_failed",
            content=f"Failed to initiate call: {result.error}",
            metadata={"error": result.error, "message": result.message},
        )

        await db.commit()

        logger.error(
            "Outbound call failed",
            call_id=str(call.id),
            error=result.error,
            to_number=request.to_number,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate call: {result.message}",
        )


@router.get("/{call_id}", response_model=CallResponse)
async def get_call(
    call_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get a call by ID."""
    service = CallService(db)
    call = await service.get(call_id)

    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )

    return CallResponse.model_validate(call)


@router.get("/{call_id}/logs", response_model=list[CallLogEntry])
async def get_call_logs(
    call_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get all log entries for a call."""
    service = CallService(db)

    # Verify call exists
    call = await service.get(call_id)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )

    logs = await service.get_logs(call_id)
    return [CallLogEntry.model_validate(log) for log in logs]


@router.get("/{call_id}/transcript", response_model=CallTranscript)
async def get_call_transcript(
    call_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get the transcript for a call."""
    service = CallService(db)

    call = await service.get(call_id)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )

    transcript_entries = await service.get_transcript(call_id)

    return CallTranscript(
        call_id=call_id,
        entries=[CallLogEntry.model_validate(e) for e in transcript_entries],
        duration_seconds=call.duration_seconds,
    )


@router.get("/{call_id}/summary", response_model=CallSummary)
async def get_call_summary(
    call_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get AI-generated summary for a call."""
    service = CallService(db)

    summary = await service.generate_summary(call_id)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )

    return summary


@router.post("/{call_id}/end")
async def end_call(
    call_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """End an active call."""
    service = CallService(db)
    twilio_client = get_twilio_client()

    call = await service.get(call_id)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )

    if call.status not in [CallStatus.QUEUED, CallStatus.RINGING, CallStatus.IN_PROGRESS]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Call is not active (status: {call.status.value})",
        )

    # Hang up via Twilio if we have a Twilio SID
    if call.twilio_call_sid:
        result = await twilio_client.hangup_call(call.twilio_call_sid)

        if not result.success:
            logger.warning(
                "Failed to hang up via Twilio",
                call_id=str(call_id),
                twilio_sid=call.twilio_call_sid,
                error=result.error,
            )

        await service.add_log(
            call_id=call_id,
            event_type="call_ended",
            content="Call ended by user",
            metadata={"twilio_result": result.success},
        )

    call = await service.end_call(call_id)
    await db.commit()

    logger.info(
        "Call ended",
        call_id=str(call_id),
        twilio_sid=call.twilio_call_sid,
    )

    return {"message": "Call ended", "call_id": str(call_id)}


@router.post("/{call_id}/transfer")
async def transfer_call(
    call_id: UUID,
    request: TransferCallRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Transfer an active call to another number."""
    service = CallService(db)
    twilio_client = get_twilio_client()

    call = await service.get(call_id)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )

    if call.status != CallStatus.IN_PROGRESS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only transfer calls that are in progress",
        )

    if not call.twilio_call_sid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Call has no Twilio SID - cannot transfer",
        )

    # Build transfer parameters
    transfer_params = TransferParams(
        call_sid=call.twilio_call_sid,
        target_number=request.target_number,
        transfer_type=TransferType.BLIND,
        announce=request.announce,
        announce_message=request.message,
    )

    # Execute transfer via Twilio
    result = await twilio_client.transfer_call_blind(transfer_params)

    if result.success:
        # Log the transfer
        await service.add_log(
            call_id=call_id,
            event_type="transfer_initiated",
            content=f"Transfer to {request.target_number}",
            metadata={
                "target_number": request.target_number,
                "announce": request.announce,
                "transfer_id": result.transfer_id,
                "transfer_type": "blind",
            },
        )

        await db.commit()

        logger.info(
            "Call transfer initiated",
            call_id=str(call_id),
            target_number=request.target_number,
            transfer_id=result.transfer_id,
        )

        return {
            "message": "Transfer initiated",
            "call_id": str(call_id),
            "target_number": request.target_number,
            "transfer_id": result.transfer_id,
        }
    else:
        # Transfer failed
        await service.add_log(
            call_id=call_id,
            event_type="transfer_failed",
            content=f"Transfer failed: {result.error}",
            metadata={
                "target_number": request.target_number,
                "error": result.error,
            },
        )

        await db.commit()

        logger.error(
            "Call transfer failed",
            call_id=str(call_id),
            target_number=request.target_number,
            error=result.error,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transfer failed: {result.message}",
        )


@router.post("/{call_id}/hold")
async def hold_call(
    call_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Put an active call on hold."""
    service = CallService(db)
    twilio_client = get_twilio_client()

    call = await service.get(call_id)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )

    if call.status != CallStatus.IN_PROGRESS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only hold calls that are in progress",
        )

    if not call.twilio_call_sid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Call has no Twilio SID - cannot hold",
        )

    result = await twilio_client.hold_call(call.twilio_call_sid)

    if result.success:
        await service.add_log(
            call_id=call_id,
            event_type="call_held",
            content="Call placed on hold",
        )
        await db.commit()

        return {"message": "Call placed on hold", "call_id": str(call_id)}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to hold call: {result.message}",
        )


@router.post("/{call_id}/resume")
async def resume_call(
    call_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Resume a call from hold."""
    service = CallService(db)
    twilio_client = get_twilio_client()
    settings = get_settings()

    call = await service.get(call_id)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )

    if not call.twilio_call_sid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Call has no Twilio SID - cannot resume",
        )

    # Build resume URL
    resume_url = f"{settings.twilio_webhook_base_url}/api/v1/webhooks/twilio/voice"

    result = await twilio_client.resume_call(call.twilio_call_sid, resume_url)

    if result.success:
        await service.add_log(
            call_id=call_id,
            event_type="call_resumed",
            content="Call resumed from hold",
        )
        await db.commit()

        return {"message": "Call resumed", "call_id": str(call_id)}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume call: {result.message}",
        )


@router.post("/{call_id}/recording/start")
async def start_call_recording(
    call_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Start recording an active call."""
    service = CallService(db)
    twilio_client = get_twilio_client()

    call = await service.get(call_id)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )

    if call.status != CallStatus.IN_PROGRESS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only record calls that are in progress",
        )

    if not call.twilio_call_sid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Call has no Twilio SID - cannot record",
        )

    result = await twilio_client.start_recording(call.twilio_call_sid)

    if result.success:
        recording_sid = result.data.get("recording_sid")

        await service.add_log(
            call_id=call_id,
            event_type="recording_started",
            content="Call recording started",
            metadata={"recording_sid": recording_sid},
        )
        await db.commit()

        return {
            "message": "Recording started",
            "call_id": str(call_id),
            "recording_sid": recording_sid,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start recording: {result.message}",
        )


@router.post("/{call_id}/recording/stop")
async def stop_call_recording(
    call_id: UUID,
    recording_sid: str = Query(..., description="Recording SID to stop"),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Stop recording an active call."""
    service = CallService(db)
    twilio_client = get_twilio_client()

    call = await service.get(call_id)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )

    if not call.twilio_call_sid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Call has no Twilio SID",
        )

    result = await twilio_client.stop_recording(call.twilio_call_sid, recording_sid)

    if result.success:
        await service.add_log(
            call_id=call_id,
            event_type="recording_stopped",
            content="Call recording stopped",
            metadata={"recording_sid": recording_sid},
        )
        await db.commit()

        return {
            "message": "Recording stopped",
            "call_id": str(call_id),
            "recording_sid": recording_sid,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop recording: {result.message}",
        )


@router.get("/{call_id}/recording", response_model=RecordingResponse)
async def get_call_recording(
    call_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get recording for a call."""
    service = CallService(db)

    call = await service.get(call_id)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )

    # Get recording URL from call metadata if available
    recording_url = call.metadata.get("recording_url") if call.metadata else None
    recording_status = "available" if recording_url else "not_available"

    return RecordingResponse(
        call_id=call_id,
        recording_url=recording_url,
        recording_duration=call.duration_seconds,
        recording_status=recording_status,
    )


@router.get("/{call_id}/status")
async def get_call_status_twilio(
    call_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get live call status from Twilio."""
    service = CallService(db)
    twilio_client = get_twilio_client()

    call = await service.get(call_id)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )

    if not call.twilio_call_sid:
        return {
            "call_id": str(call_id),
            "status": call.status.value,
            "source": "database",
        }

    result = await twilio_client.get_call_status(call.twilio_call_sid)

    if result.success:
        return {
            "call_id": str(call_id),
            "twilio_call_sid": call.twilio_call_sid,
            "status": result.status,
            "source": "twilio",
            "details": result.data,
        }
    else:
        return {
            "call_id": str(call_id),
            "status": call.status.value,
            "source": "database",
            "twilio_error": result.error,
        }


@router.get("/stats/overview")
async def get_call_stats(
    agent_id: Optional[UUID] = Query(None, description="Filter by agent"),
    from_date: Optional[datetime] = Query(None, description="Filter from date"),
    to_date: Optional[datetime] = Query(None, description="Filter to date"),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get call statistics."""
    service = CallService(db)
    return await service.get_call_stats(
        agent_id=agent_id,
        from_date=from_date,
        to_date=to_date,
    )

"""Call API routes."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

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

    # Create call record
    call = await service.create(
        agent_id=request.agent_id,
        direction=CallDirection.OUTBOUND,
        from_number=request.from_number,
        to_number=request.to_number,
        metadata=request.metadata,
    )

    await db.commit()

    # TODO: Integrate with Telephony Gateway to actually place the call

    return OutboundCallResponse(
        call_id=call.id,
        session_id=call.session_id,
        status=call.status,
        message="Call queued for dialing",
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

    # TODO: Send end signal to Telephony Gateway

    call = await service.end_call(call_id)
    await db.commit()

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

    # TODO: Integrate with Telephony Gateway to transfer

    # Log the transfer
    await service.add_log(
        call_id=call_id,
        event_type="transfer",
        content=f"Transfer to {request.target_number}",
        metadata={"target_number": request.target_number, "announce": request.announce},
    )

    await db.commit()

    return {
        "message": "Transfer initiated",
        "call_id": str(call_id),
        "target_number": request.target_number,
    }


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

    # TODO: Integrate with recording storage

    return RecordingResponse(
        call_id=call_id,
        recording_url=None,
        recording_duration=call.duration_seconds,
        recording_status="not_available",
    )


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

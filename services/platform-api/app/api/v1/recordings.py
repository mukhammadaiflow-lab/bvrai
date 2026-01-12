"""
Recordings API Routes

Handles:
- Call recording management
- Transcription retrieval
- Recording download
- Storage management
"""

from typing import Optional, List
from datetime import datetime, date
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_db_session,
    get_current_user,
    get_current_tenant,
    UserContext,
    TenantContext,
    require_permissions,
    get_pagination,
    PaginationParams,
)

router = APIRouter(prefix="/recordings")


# ============================================================================
# Schemas
# ============================================================================

class RecordingStatus(str, Enum):
    """Recording status."""
    RECORDING = "recording"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class TranscriptionStatus(str, Enum):
    """Transcription status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptSegment(BaseModel):
    """Transcript segment."""
    speaker: str
    text: str
    start_time: float
    end_time: float
    confidence: float


class TranscriptionResponse(BaseModel):
    """Transcription response."""
    id: str
    recording_id: str
    status: TranscriptionStatus
    full_text: str
    segments: List[TranscriptSegment]
    language: str
    word_count: int
    speaker_count: int
    duration_seconds: float
    created_at: datetime
    completed_at: Optional[datetime]


class RecordingResponse(BaseModel):
    """Recording response."""
    id: str
    call_id: str
    agent_id: Optional[str]
    status: RecordingStatus
    duration_seconds: float
    file_size_bytes: int
    format: str
    sample_rate: int
    channels: int
    download_url: Optional[str]
    transcription_status: TranscriptionStatus
    transcription_id: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class RecordingListResponse(BaseModel):
    """List response with pagination."""
    recordings: List[RecordingResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class RecordingStats(BaseModel):
    """Recording statistics."""
    total_recordings: int
    total_duration_hours: float
    total_size_gb: float
    transcribed_count: int
    transcription_rate: float
    average_duration_minutes: float


class TranscribeRequest(BaseModel):
    """Manual transcription request."""
    provider: Optional[str] = None
    language: Optional[str] = "en"


class ExportFormat(str, Enum):
    """Export formats."""
    SRT = "srt"
    VTT = "vtt"
    TXT = "txt"
    JSON = "json"


# ============================================================================
# Recordings CRUD
# ============================================================================

@router.get("", response_model=RecordingListResponse)
async def list_recordings(
    call_id: Optional[UUID] = None,
    agent_id: Optional[UUID] = None,
    status_filter: Optional[RecordingStatus] = Query(None, alias="status"),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    has_transcription: Optional[bool] = None,
    pagination: PaginationParams = Depends(get_pagination),
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """List all recordings for the tenant."""
    from app.recording import get_recording_service

    service = get_recording_service()

    filters = {
        "tenant_id": tenant.tenant_id,
        "call_id": str(call_id) if call_id else None,
        "agent_id": str(agent_id) if agent_id else None,
        "status": status_filter.value if status_filter else None,
        "start_date": start_date,
        "end_date": end_date,
        "has_transcription": has_transcription,
    }

    recordings = service.list_by_tenant(
        tenant.tenant_id,
        limit=pagination.limit,
        offset=pagination.offset,
    )

    total = len(recordings)

    return RecordingListResponse(
        recordings=[
            RecordingResponse(
                id=r.recording_id,
                call_id=r.call_id,
                agent_id=r.agent_id,
                status=RecordingStatus(r.status.value),
                duration_seconds=r.duration_seconds,
                file_size_bytes=r.file_size_bytes,
                format=r.format.value,
                sample_rate=r.sample_rate,
                channels=r.channels,
                download_url=r.storage_url,
                transcription_status=TranscriptionStatus(r.transcription_status.value),
                transcription_id=r.transcription_id,
                created_at=r.created_at,
                completed_at=r.stopped_at,
            )
            for r in recordings
        ],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=(total + pagination.page_size - 1) // pagination.page_size,
    )


@router.get("/stats", response_model=RecordingStats)
async def get_recording_stats(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get recording statistics."""
    from app.recording import get_recording_service

    service = get_recording_service()
    stats = service.get_stats()

    return RecordingStats(
        total_recordings=stats["total_recordings"],
        total_duration_hours=stats["total_duration_seconds"] / 3600,
        total_size_gb=stats["total_size_bytes"] / (1024 ** 3),
        transcribed_count=stats["total_transcriptions"],
        transcription_rate=stats["transcription_rate"],
        average_duration_minutes=stats["total_duration_seconds"] / max(stats["total_recordings"], 1) / 60,
    )


@router.get("/{recording_id}", response_model=RecordingResponse)
async def get_recording(
    recording_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get a recording by ID."""
    from app.recording import get_recording_service

    service = get_recording_service()
    recording = service.get_recording(str(recording_id))

    if not recording or recording.tenant_id != tenant.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found",
        )

    return RecordingResponse(
        id=recording.recording_id,
        call_id=recording.call_id,
        agent_id=recording.agent_id,
        status=RecordingStatus(recording.status.value),
        duration_seconds=recording.duration_seconds,
        file_size_bytes=recording.file_size_bytes,
        format=recording.format.value,
        sample_rate=recording.sample_rate,
        channels=recording.channels,
        download_url=recording.storage_url,
        transcription_status=TranscriptionStatus(recording.transcription_status.value),
        transcription_id=recording.transcription_id,
        created_at=recording.created_at,
        completed_at=recording.stopped_at,
    )


@router.delete("/{recording_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_recording(
    recording_id: UUID,
    user: UserContext = Depends(require_permissions("recordings:delete")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a recording."""
    from app.recording import get_recording_service

    service = get_recording_service()
    recording = service.get_recording(str(recording_id))

    if not recording or recording.tenant_id != tenant.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found",
        )

    await service.delete_recording(str(recording_id))


# ============================================================================
# Download
# ============================================================================

@router.get("/{recording_id}/download")
async def download_recording(
    recording_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Download recording audio file.

    Returns a signed URL or streams the audio file directly.
    """
    from app.recording import get_recording_service, get_storage_manager

    service = get_recording_service()
    recording = service.get_recording(str(recording_id))

    if not recording or recording.tenant_id != tenant.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found",
        )

    if not recording.storage_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording file not available",
        )

    storage = get_storage_manager()

    # Generate signed URL for cloud storage
    if recording.storage_url.startswith("s3://") or recording.storage_url.startswith("gs://"):
        signed_url = await storage.get_signed_url(
            recording.storage_path,
            expires_in=3600,
        )
        return {"download_url": signed_url}

    # Stream local file
    async def stream_file():
        with open(recording.storage_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    return StreamingResponse(
        stream_file(),
        media_type=f"audio/{recording.format.value}",
        headers={
            "Content-Disposition": f"attachment; filename={recording.recording_id}.{recording.format.value}",
            "Content-Length": str(recording.file_size_bytes),
        },
    )


# ============================================================================
# Transcriptions
# ============================================================================

@router.get("/{recording_id}/transcription", response_model=TranscriptionResponse)
async def get_transcription(
    recording_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get transcription for a recording."""
    from app.recording import get_recording_service

    service = get_recording_service()
    recording = service.get_recording(str(recording_id))

    if not recording or recording.tenant_id != tenant.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found",
        )

    transcription = service.get_recording_transcription(str(recording_id))

    if not transcription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription not available",
        )

    return TranscriptionResponse(
        id=transcription.transcription_id,
        recording_id=transcription.recording_id,
        status=TranscriptionStatus(transcription.status.value),
        full_text=transcription.full_text,
        segments=[
            TranscriptSegment(
                speaker=s.speaker,
                text=s.text,
                start_time=s.start_time,
                end_time=s.end_time,
                confidence=s.confidence,
            )
            for s in transcription.segments
        ],
        language=transcription.language,
        word_count=transcription.word_count,
        speaker_count=transcription.speaker_count,
        duration_seconds=transcription.duration_seconds,
        created_at=transcription.created_at,
        completed_at=transcription.completed_at,
    )


@router.post("/{recording_id}/transcribe", response_model=TranscriptionResponse)
async def transcribe_recording(
    recording_id: UUID,
    data: TranscribeRequest,
    user: UserContext = Depends(require_permissions("recordings:transcribe")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Request transcription for a recording.

    Starts a new transcription job if one doesn't exist.
    """
    from app.recording import get_recording_service

    service = get_recording_service()
    recording = service.get_recording(str(recording_id))

    if not recording or recording.tenant_id != tenant.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found",
        )

    # Check if already transcribed
    existing = service.get_recording_transcription(str(recording_id))
    if existing and existing.status.value == "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Recording already transcribed",
        )

    transcription = await service.transcribe(str(recording_id))

    return TranscriptionResponse(
        id=transcription.transcription_id,
        recording_id=transcription.recording_id,
        status=TranscriptionStatus(transcription.status.value),
        full_text=transcription.full_text,
        segments=[
            TranscriptSegment(
                speaker=s.speaker,
                text=s.text,
                start_time=s.start_time,
                end_time=s.end_time,
                confidence=s.confidence,
            )
            for s in transcription.segments
        ],
        language=transcription.language,
        word_count=transcription.word_count,
        speaker_count=transcription.speaker_count,
        duration_seconds=transcription.duration_seconds,
        created_at=transcription.created_at,
        completed_at=transcription.completed_at,
    )


@router.get("/{recording_id}/transcription/export")
async def export_transcription(
    recording_id: UUID,
    format: ExportFormat = Query(ExportFormat.TXT),
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Export transcription in various formats.

    Supports: SRT, VTT, TXT, JSON
    """
    from app.recording import get_recording_service
    from app.recording.transcription import Transcript

    service = get_recording_service()
    recording = service.get_recording(str(recording_id))

    if not recording or recording.tenant_id != tenant.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found",
        )

    transcription = service.get_recording_transcription(str(recording_id))

    if not transcription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription not available",
        )

    # Build transcript object for export
    from app.recording.transcription import TranscriptSegment as TSegment
    transcript = Transcript(
        recording_id=str(recording_id),
        segments=[
            TSegment(
                speaker=s.speaker,
                text=s.text,
                start_time=s.start_time,
                end_time=s.end_time,
                confidence=s.confidence,
            )
            for s in transcription.segments
        ],
        full_text=transcription.full_text,
        duration=transcription.duration_seconds,
    )

    # Export based on format
    if format == ExportFormat.SRT:
        content = transcript.to_srt()
        media_type = "application/x-subrip"
        ext = "srt"
    elif format == ExportFormat.VTT:
        content = transcript.to_vtt()
        media_type = "text/vtt"
        ext = "vtt"
    elif format == ExportFormat.JSON:
        content = transcript.to_dict()
        import json
        content = json.dumps(content, indent=2)
        media_type = "application/json"
        ext = "json"
    else:  # TXT
        content = transcript.full_text
        media_type = "text/plain"
        ext = "txt"

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={recording_id}.{ext}",
        },
    )


# ============================================================================
# Bulk Operations
# ============================================================================

@router.post("/bulk/transcribe")
async def bulk_transcribe(
    recording_ids: List[UUID],
    user: UserContext = Depends(require_permissions("recordings:transcribe")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Start transcription for multiple recordings."""
    from app.recording import get_recording_service

    service = get_recording_service()

    results = []
    for rid in recording_ids:
        recording = service.get_recording(str(rid))
        if recording and recording.tenant_id == tenant.tenant_id:
            try:
                await service.transcribe(str(rid))
                results.append({"recording_id": str(rid), "status": "queued"})
            except Exception as e:
                results.append({"recording_id": str(rid), "status": "error", "error": str(e)})
        else:
            results.append({"recording_id": str(rid), "status": "not_found"})

    return {"results": results}


@router.delete("/bulk")
async def bulk_delete(
    recording_ids: List[UUID],
    user: UserContext = Depends(require_permissions("recordings:delete")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete multiple recordings."""
    from app.recording import get_recording_service

    service = get_recording_service()

    deleted = 0
    for rid in recording_ids:
        recording = service.get_recording(str(rid))
        if recording and recording.tenant_id == tenant.tenant_id:
            await service.delete_recording(str(rid))
            deleted += 1

    return {"deleted": deleted}

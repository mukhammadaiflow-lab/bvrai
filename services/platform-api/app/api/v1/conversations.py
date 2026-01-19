"""
Conversations API Routes

Handles:
- Conversation history
- Message management
- Context retrieval
- Conversation analytics
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, date
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_db_session,
    get_current_user,
    get_current_tenant,
    UserContext,
    TenantContext,
    get_pagination,
    PaginationParams,
)

router = APIRouter(prefix="/conversations")


# ============================================================================
# Schemas
# ============================================================================

class ConversationStatus(str, Enum):
    """Conversation status."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    TRANSFERRED = "transferred"


class MessageRole(str, Enum):
    """Message role."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class SentimentType(str, Enum):
    """Sentiment type."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class MessageResponse(BaseModel):
    """Message response."""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}
    sentiment: Optional[SentimentType] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None


class ConversationResponse(BaseModel):
    """Conversation response."""
    id: str
    call_id: str
    agent_id: str
    agent_name: Optional[str]
    status: ConversationStatus
    started_at: datetime
    ended_at: Optional[datetime]
    duration_seconds: float
    message_count: int
    user_message_count: int
    agent_message_count: int
    summary: Optional[str]
    sentiment: Optional[SentimentType]
    intents: List[str]
    entities: Dict[str, Any]
    outcome: Optional[str]
    metadata: Dict[str, Any]

    class Config:
        from_attributes = True


class ConversationDetailResponse(ConversationResponse):
    """Detailed conversation with messages."""
    messages: List[MessageResponse]


class ConversationListResponse(BaseModel):
    """List response with pagination."""
    conversations: List[ConversationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ConversationStats(BaseModel):
    """Conversation statistics."""
    total_conversations: int
    completed_conversations: int
    abandoned_conversations: int
    average_duration_seconds: float
    average_messages_per_conversation: float
    sentiment_distribution: Dict[str, int]
    top_intents: List[Dict[str, Any]]
    resolution_rate: float


class ConversationSearch(BaseModel):
    """Search parameters."""
    query: Optional[str] = None
    agent_id: Optional[UUID] = None
    status: Optional[ConversationStatus] = None
    sentiment: Optional[SentimentType] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    min_duration: Optional[int] = None
    max_duration: Optional[int] = None
    has_recording: Optional[bool] = None


class ExportFormat(str, Enum):
    """Export formats."""
    JSON = "json"
    CSV = "csv"
    TXT = "txt"


# ============================================================================
# List & Search
# ============================================================================

@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    agent_id: Optional[UUID] = None,
    status_filter: Optional[ConversationStatus] = Query(None, alias="status"),
    sentiment: Optional[SentimentType] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    pagination: PaginationParams = Depends(get_pagination),
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """List all conversations."""
    from app.conversation import ConversationManager

    manager = ConversationManager()

    conversations, total = await manager.list_conversations(
        tenant_id=tenant.tenant_id,
        agent_id=str(agent_id) if agent_id else None,
        status=status_filter.value if status_filter else None,
        sentiment=sentiment.value if sentiment else None,
        start_date=start_date,
        end_date=end_date,
        offset=pagination.offset,
        limit=pagination.limit,
    )

    return ConversationListResponse(
        conversations=[
            ConversationResponse(
                id=c.conversation_id,
                call_id=c.call_id,
                agent_id=c.agent_id,
                agent_name=getattr(c, 'agent_name', None),
                status=ConversationStatus(c.status),
                started_at=c.started_at,
                ended_at=c.ended_at,
                duration_seconds=c.duration_seconds,
                message_count=c.message_count,
                user_message_count=c.user_message_count,
                agent_message_count=c.agent_message_count,
                summary=c.summary,
                sentiment=SentimentType(c.sentiment) if c.sentiment else None,
                intents=c.intents or [],
                entities=c.entities or {},
                outcome=c.outcome,
                metadata=c.metadata or {},
            )
            for c in conversations
        ],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=(total + pagination.page_size - 1) // pagination.page_size,
    )


@router.post("/search", response_model=ConversationListResponse)
async def search_conversations(
    search: ConversationSearch,
    pagination: PaginationParams = Depends(get_pagination),
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Search conversations with advanced filters."""
    from app.conversation import ConversationManager

    manager = ConversationManager()

    conversations, total = await manager.search_conversations(
        tenant_id=tenant.tenant_id,
        query=search.query,
        agent_id=str(search.agent_id) if search.agent_id else None,
        status=search.status.value if search.status else None,
        sentiment=search.sentiment.value if search.sentiment else None,
        start_date=search.start_date,
        end_date=search.end_date,
        min_duration=search.min_duration,
        max_duration=search.max_duration,
        offset=pagination.offset,
        limit=pagination.limit,
    )

    return ConversationListResponse(
        conversations=[
            ConversationResponse(
                id=c.conversation_id,
                call_id=c.call_id,
                agent_id=c.agent_id,
                agent_name=getattr(c, 'agent_name', None),
                status=ConversationStatus(c.status),
                started_at=c.started_at,
                ended_at=c.ended_at,
                duration_seconds=c.duration_seconds,
                message_count=c.message_count,
                user_message_count=c.user_message_count,
                agent_message_count=c.agent_message_count,
                summary=c.summary,
                sentiment=SentimentType(c.sentiment) if c.sentiment else None,
                intents=c.intents or [],
                entities=c.entities or {},
                outcome=c.outcome,
                metadata=c.metadata or {},
            )
            for c in conversations
        ],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=(total + pagination.page_size - 1) // pagination.page_size,
    )


@router.get("/stats", response_model=ConversationStats)
async def get_conversation_stats(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    agent_id: Optional[UUID] = None,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get conversation statistics."""
    from app.conversation import ConversationManager

    manager = ConversationManager()

    stats = await manager.get_stats(
        tenant_id=tenant.tenant_id,
        start_date=start_date,
        end_date=end_date,
        agent_id=str(agent_id) if agent_id else None,
    )

    return ConversationStats(**stats)


# ============================================================================
# Single Conversation
# ============================================================================

@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get conversation with messages."""
    from app.conversation import ConversationManager

    manager = ConversationManager()

    conversation = await manager.get_conversation(
        conversation_id=str(conversation_id),
        tenant_id=tenant.tenant_id,
    )

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    messages = await manager.get_messages(str(conversation_id))

    return ConversationDetailResponse(
        id=conversation.conversation_id,
        call_id=conversation.call_id,
        agent_id=conversation.agent_id,
        agent_name=getattr(conversation, 'agent_name', None),
        status=ConversationStatus(conversation.status),
        started_at=conversation.started_at,
        ended_at=conversation.ended_at,
        duration_seconds=conversation.duration_seconds,
        message_count=conversation.message_count,
        user_message_count=conversation.user_message_count,
        agent_message_count=conversation.agent_message_count,
        summary=conversation.summary,
        sentiment=SentimentType(conversation.sentiment) if conversation.sentiment else None,
        intents=conversation.intents or [],
        entities=conversation.entities or {},
        outcome=conversation.outcome,
        metadata=conversation.metadata or {},
        messages=[
            MessageResponse(
                id=m.message_id,
                role=MessageRole(m.role),
                content=m.content,
                timestamp=m.timestamp,
                metadata=m.metadata or {},
                sentiment=SentimentType(m.sentiment) if m.sentiment else None,
                intent=m.intent,
                confidence=m.confidence,
            )
            for m in messages
        ],
    )


@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get conversation messages."""
    from app.conversation import ConversationManager

    manager = ConversationManager()

    # Verify conversation belongs to tenant
    conversation = await manager.get_conversation(
        conversation_id=str(conversation_id),
        tenant_id=tenant.tenant_id,
    )

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    messages = await manager.get_messages(str(conversation_id))

    return [
        MessageResponse(
            id=m.message_id,
            role=MessageRole(m.role),
            content=m.content,
            timestamp=m.timestamp,
            metadata=m.metadata or {},
            sentiment=SentimentType(m.sentiment) if m.sentiment else None,
            intent=m.intent,
            confidence=m.confidence,
        )
        for m in messages
    ]


@router.get("/{conversation_id}/summary")
async def get_conversation_summary(
    conversation_id: UUID,
    regenerate: bool = False,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get or generate conversation summary."""
    from app.conversation import ConversationManager

    manager = ConversationManager()

    conversation = await manager.get_conversation(
        conversation_id=str(conversation_id),
        tenant_id=tenant.tenant_id,
    )

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    if regenerate or not conversation.summary:
        summary = await manager.generate_summary(str(conversation_id))
    else:
        summary = conversation.summary

    return {
        "conversation_id": str(conversation_id),
        "summary": summary,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/{conversation_id}/analysis")
async def analyze_conversation(
    conversation_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get detailed conversation analysis."""
    from app.conversation import ConversationManager

    manager = ConversationManager()

    conversation = await manager.get_conversation(
        conversation_id=str(conversation_id),
        tenant_id=tenant.tenant_id,
    )

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    analysis = await manager.analyze_conversation(str(conversation_id))

    return {
        "conversation_id": str(conversation_id),
        "analysis": analysis,
    }


# ============================================================================
# Export
# ============================================================================

@router.get("/{conversation_id}/export")
async def export_conversation(
    conversation_id: UUID,
    format: ExportFormat = Query(ExportFormat.JSON),
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Export conversation in various formats."""
    from fastapi.responses import Response
    from app.conversation import ConversationManager
    import json

    manager = ConversationManager()

    conversation = await manager.get_conversation(
        conversation_id=str(conversation_id),
        tenant_id=tenant.tenant_id,
    )

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    messages = await manager.get_messages(str(conversation_id))

    if format == ExportFormat.JSON:
        content = json.dumps({
            "conversation_id": str(conversation_id),
            "agent_id": conversation.agent_id,
            "started_at": conversation.started_at.isoformat(),
            "ended_at": conversation.ended_at.isoformat() if conversation.ended_at else None,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in messages
            ],
        }, indent=2)
        media_type = "application/json"
        ext = "json"

    elif format == ExportFormat.CSV:
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "role", "content"])
        for m in messages:
            writer.writerow([m.timestamp.isoformat(), m.role, m.content])
        content = output.getvalue()
        media_type = "text/csv"
        ext = "csv"

    else:  # TXT
        lines = []
        for m in messages:
            lines.append(f"[{m.timestamp.strftime('%H:%M:%S')}] {m.role.upper()}: {m.content}")
        content = "\n".join(lines)
        media_type = "text/plain"
        ext = "txt"

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=conversation_{conversation_id}.{ext}",
        },
    )


# ============================================================================
# Batch Operations
# ============================================================================

@router.post("/batch/analyze")
async def batch_analyze(
    conversation_ids: List[UUID],
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Analyze multiple conversations."""
    from app.conversation import ConversationManager

    manager = ConversationManager()

    results = []
    for cid in conversation_ids:
        try:
            analysis = await manager.analyze_conversation(str(cid))
            results.append({
                "conversation_id": str(cid),
                "status": "success",
                "analysis": analysis,
            })
        except Exception as e:
            results.append({
                "conversation_id": str(cid),
                "status": "error",
                "error": str(e),
            })

    return {"results": results}

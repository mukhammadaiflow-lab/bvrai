"""
LLM API Routes

Handles:
- LLM completions
- Chat completions
- Model management
- Token usage
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_db_session,
    get_current_user,
    get_current_tenant,
    UserContext,
    TenantContext,
    RateLimitDep,
)

router = APIRouter(prefix="/llm")


# ============================================================================
# Schemas
# ============================================================================

class LLMProvider(str, Enum):
    """LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    TOGETHER = "together"
    MISTRAL = "mistral"
    AZURE = "azure"


class MessageRole(str, Enum):
    """Message role."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class Message(BaseModel):
    """Chat message."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class FunctionDefinition(BaseModel):
    """Function definition."""
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolDefinition(BaseModel):
    """Tool definition."""
    type: str = "function"
    function: FunctionDefinition


class CompletionRequest(BaseModel):
    """Completion request."""
    prompt: str = Field(..., min_length=1, max_length=100000)
    model: str = "gpt-4o"
    provider: Optional[LLMProvider] = None
    max_tokens: int = Field(1000, ge=1, le=32000)
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(1.0, ge=0, le=1)
    stop: Optional[List[str]] = None
    stream: bool = False


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""
    messages: List[Message]
    model: str = "gpt-4o"
    provider: Optional[LLMProvider] = None
    max_tokens: int = Field(1000, ge=1, le=32000)
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(1.0, ge=0, le=1)
    stop: Optional[List[str]] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[str] = None
    stream: bool = False


class CompletionResponse(BaseModel):
    """Completion response."""
    id: str
    content: str
    model: str
    provider: str
    finish_reason: str
    usage: Dict[str, int]
    created_at: datetime


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str
    message: Message
    model: str
    provider: str
    finish_reason: str
    usage: Dict[str, int]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    created_at: datetime


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    name: str
    provider: LLMProvider
    context_length: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    capabilities: List[str]
    is_available: bool


class UsageResponse(BaseModel):
    """Usage response."""
    total_requests: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    by_model: Dict[str, Dict[str, Any]]
    by_day: List[Dict[str, Any]]


# ============================================================================
# Rate Limiters
# ============================================================================

completion_rate_limit = RateLimitDep(requests=60, window=60, scope="llm_completion")


# ============================================================================
# Completions
# ============================================================================

@router.post("/completions", response_model=CompletionResponse)
async def create_completion(
    data: CompletionRequest,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    _rate_limit: None = Depends(completion_rate_limit),
):
    """Create a text completion."""
    from app.llm import LLMEngine

    engine = LLMEngine()

    if data.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use /completions/stream for streaming",
        )

    try:
        result = await engine.complete(
            prompt=data.prompt,
            model=data.model,
            provider=data.provider.value if data.provider else None,
            max_tokens=data.max_tokens,
            temperature=data.temperature,
            top_p=data.top_p,
            stop=data.stop,
        )

        return CompletionResponse(
            id=result.id,
            content=result.content,
            model=result.model,
            provider=result.provider,
            finish_reason=result.finish_reason,
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            },
            created_at=datetime.utcnow(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Completion failed: {str(e)}",
        )


@router.post("/completions/stream")
async def stream_completion(
    data: CompletionRequest,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    _rate_limit: None = Depends(completion_rate_limit),
):
    """Stream a text completion."""
    from app.llm import LLMEngine
    import json

    engine = LLMEngine()

    async def generate():
        async for chunk in engine.stream_complete(
            prompt=data.prompt,
            model=data.model,
            provider=data.provider.value if data.provider else None,
            max_tokens=data.max_tokens,
            temperature=data.temperature,
            top_p=data.top_p,
            stop=data.stop,
        ):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


# ============================================================================
# Chat Completions
# ============================================================================

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    data: ChatCompletionRequest,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    _rate_limit: None = Depends(completion_rate_limit),
):
    """Create a chat completion."""
    from app.llm import LLMEngine

    engine = LLMEngine()

    if data.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use /chat/completions/stream for streaming",
        )

    try:
        messages = [
            {"role": m.role.value, "content": m.content, "name": m.name}
            for m in data.messages
        ]

        tools = None
        if data.tools:
            tools = [t.model_dump() for t in data.tools]

        result = await engine.chat(
            messages=messages,
            model=data.model,
            provider=data.provider.value if data.provider else None,
            max_tokens=data.max_tokens,
            temperature=data.temperature,
            top_p=data.top_p,
            stop=data.stop,
            tools=tools,
            tool_choice=data.tool_choice,
        )

        return ChatCompletionResponse(
            id=result.id,
            message=Message(
                role=MessageRole(result.message["role"]),
                content=result.message.get("content", ""),
                tool_calls=result.message.get("tool_calls"),
            ),
            model=result.model,
            provider=result.provider,
            finish_reason=result.finish_reason,
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            },
            tool_calls=result.tool_calls,
            created_at=datetime.utcnow(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {str(e)}",
        )


@router.post("/chat/completions/stream")
async def stream_chat_completion(
    data: ChatCompletionRequest,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    _rate_limit: None = Depends(completion_rate_limit),
):
    """Stream a chat completion."""
    from app.llm import LLMEngine
    import json

    engine = LLMEngine()

    messages = [
        {"role": m.role.value, "content": m.content, "name": m.name}
        for m in data.messages
    ]

    tools = None
    if data.tools:
        tools = [t.model_dump() for t in data.tools]

    async def generate():
        async for chunk in engine.stream_chat(
            messages=messages,
            model=data.model,
            provider=data.provider.value if data.provider else None,
            max_tokens=data.max_tokens,
            temperature=data.temperature,
            top_p=data.top_p,
            stop=data.stop,
            tools=tools,
            tool_choice=data.tool_choice,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


# ============================================================================
# Models
# ============================================================================

@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    provider: Optional[LLMProvider] = None,
    capability: Optional[str] = None,
    user: UserContext = Depends(get_current_user),
):
    """List available models."""
    from app.llm import LLMEngine

    engine = LLMEngine()
    models = engine.list_models(
        provider=provider.value if provider else None,
        capability=capability,
    )

    return [
        ModelInfo(
            id=m["id"],
            name=m["name"],
            provider=LLMProvider(m["provider"]),
            context_length=m["context_length"],
            input_cost_per_1k=m["input_cost_per_1k"],
            output_cost_per_1k=m["output_cost_per_1k"],
            capabilities=m["capabilities"],
            is_available=m["is_available"],
        )
        for m in models
    ]


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Get model details."""
    from app.llm import LLMEngine

    engine = LLMEngine()
    model = engine.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    return ModelInfo(
        id=model["id"],
        name=model["name"],
        provider=LLMProvider(model["provider"]),
        context_length=model["context_length"],
        input_cost_per_1k=model["input_cost_per_1k"],
        output_cost_per_1k=model["output_cost_per_1k"],
        capabilities=model["capabilities"],
        is_available=model["is_available"],
    )


# ============================================================================
# Usage
# ============================================================================

@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    model: Optional[str] = None,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get LLM usage statistics."""
    from app.llm import LLMEngine

    engine = LLMEngine()

    usage = await engine.get_usage(
        tenant_id=tenant.tenant_id,
        start_date=start_date,
        end_date=end_date,
        model=model,
    )

    return UsageResponse(
        total_requests=usage["total_requests"],
        total_tokens=usage["total_tokens"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        estimated_cost=usage["estimated_cost"],
        by_model=usage["by_model"],
        by_day=usage["by_day"],
    )


# ============================================================================
# Embeddings
# ============================================================================

class EmbeddingRequest(BaseModel):
    """Embedding request."""
    input: List[str] = Field(..., min_items=1, max_items=100)
    model: str = "text-embedding-3-small"
    provider: Optional[LLMProvider] = None


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    data: EmbeddingRequest,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Create embeddings for text."""
    from app.knowledge import EmbeddingService

    service = EmbeddingService()

    try:
        embeddings = await service.embed_batch(
            texts=data.input,
            model=data.model,
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=data.model,
            usage={
                "prompt_tokens": sum(len(t.split()) for t in data.input),
                "total_tokens": sum(len(t.split()) for t in data.input),
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {str(e)}",
        )

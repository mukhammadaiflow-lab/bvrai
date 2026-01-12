"""AI Orchestrator API - LLM and RAG integration."""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import get_settings
from app.llm import OpenAIAdapter, AnthropicAdapter, MockLLMAdapter, LLMAdapter
from app.llm.base import Message, Tool
from app.rag import RAGPipeline, AgentRAGPipeline, RAGResult

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
settings = get_settings()

# Global adapters
llm_adapter: Optional[LLMAdapter] = None
rag_pipeline: Optional[RAGPipeline] = None
agent_rag_pipelines: dict[str, AgentRAGPipeline] = {}


def create_adapter() -> LLMAdapter:
    """Create LLM adapter based on configuration."""
    provider = settings.llm_provider

    if provider == "openai":
        return OpenAIAdapter()
    elif provider == "anthropic":
        return AnthropicAdapter()
    elif provider == "mock":
        return MockLLMAdapter()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    global llm_adapter, rag_pipeline

    logger.info(
        "Starting AI Orchestrator",
        provider=settings.llm_provider,
        port=settings.port,
    )

    llm_adapter = create_adapter()

    # Initialize RAG pipeline
    if settings.rag_enabled:
        rag_pipeline = RAGPipeline()
        logger.info("RAG pipeline initialized")

    yield

    logger.info("Shutting down AI Orchestrator")
    if llm_adapter:
        await llm_adapter.close()
    if rag_pipeline:
        await rag_pipeline.close()
    for pipeline in agent_rag_pipelines.values():
        await pipeline.close()


app = FastAPI(
    title="AI Orchestrator",
    description="LLM and RAG integration for Builder Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class HealthResponse(BaseModel):
    status: str
    service: str
    provider: str


class MessageInput(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ToolInput(BaseModel):
    name: str
    description: str
    parameters: dict


class GenerateRequest(BaseModel):
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    transcript: Optional[str] = None  # Current user input
    messages: list[MessageInput] = []  # Conversation history
    system_prompt: Optional[str] = None
    tools: list[ToolInput] = []
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class FunctionCallOutput(BaseModel):
    id: Optional[str] = None
    name: str
    arguments: dict


class GenerateResponse(BaseModel):
    text: str
    function_calls: list[FunctionCallOutput] = []
    finish_reason: str = "stop"


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check."""
    return HealthResponse(
        status="healthy",
        service="ai-orchestrator",
        provider=settings.llm_provider,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate a response from the LLM.

    For voice conversations, this takes the current transcript
    and conversation history, then returns the agent's response.
    """
    if not llm_adapter:
        raise HTTPException(status_code=503, detail="LLM adapter not initialized")

    # Build messages
    messages = []

    # Add conversation history
    for msg in request.messages:
        messages.append(
            Message(
                role=msg.role,
                content=msg.content,
                name=msg.name,
            )
        )

    # Add current transcript as user message
    if request.transcript:
        messages.append(Message(role="user", content=request.transcript))

    # Convert tools
    tools = None
    if request.tools:
        tools = [
            Tool(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
            )
            for t in request.tools
        ]

    # Generate response
    kwargs = {}
    if request.max_tokens:
        kwargs["max_tokens"] = request.max_tokens
    if request.temperature:
        kwargs["temperature"] = request.temperature

    try:
        response = await llm_adapter.generate(
            messages=messages,
            system_prompt=request.system_prompt,
            tools=tools,
            **kwargs,
        )

        return GenerateResponse(
            text=response.text,
            function_calls=[
                FunctionCallOutput(
                    id=fc.id,
                    name=fc.name,
                    arguments=fc.arguments,
                )
                for fc in response.function_calls
            ],
            finish_reason=response.finish_reason,
        )

    except Exception as e:
        logger.error("Generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    Stream a response from the LLM.

    Returns Server-Sent Events with partial responses.
    """
    if not llm_adapter:
        raise HTTPException(status_code=503, detail="LLM adapter not initialized")

    # Build messages
    messages = []
    for msg in request.messages:
        messages.append(
            Message(role=msg.role, content=msg.content, name=msg.name)
        )

    if request.transcript:
        messages.append(Message(role="user", content=request.transcript))

    # Convert tools
    tools = None
    if request.tools:
        tools = [
            Tool(name=t.name, description=t.description, parameters=t.parameters)
            for t in request.tools
        ]

    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events."""
        try:
            async for response in llm_adapter.generate_stream(
                messages=messages,
                system_prompt=request.system_prompt,
                tools=tools,
            ):
                if response.is_partial:
                    # Send text chunk
                    yield f"data: {response.text}\n\n"
                elif response.is_complete:
                    # Send done event
                    yield f"event: done\ndata: {response.finish_reason}\n\n"

        except Exception as e:
            logger.error("Streaming error", error=str(e))
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# Voice-specific endpoint
class VoiceTurnRequest(BaseModel):
    session_id: str
    agent_id: str
    transcript: str
    conversation_history: list[MessageInput] = []
    agent_config: Optional[dict] = None


class VoiceTurnResponse(BaseModel):
    text: str  # Text to speak
    action: Optional[str] = None  # Action to take (transfer, hangup, etc.)
    action_params: Optional[dict] = None


@app.post("/voice/turn", response_model=VoiceTurnResponse)
async def handle_voice_turn(request: VoiceTurnRequest):
    """
    Handle a single voice conversation turn.

    Takes user transcript and returns agent response optimized for speech.
    """
    if not llm_adapter:
        raise HTTPException(status_code=503, detail="LLM adapter not initialized")

    # Build messages from history
    messages = [
        Message(role=msg.role, content=msg.content)
        for msg in request.conversation_history
    ]

    # Add current transcript
    messages.append(Message(role="user", content=request.transcript))

    # Get system prompt from agent config or use default
    system_prompt = None
    if request.agent_config:
        system_prompt = request.agent_config.get("system_prompt")

    # Define voice-specific tools
    tools = [
        Tool(
            name="transfer_call",
            description="Transfer the call to a human agent",
            parameters={
                "type": "object",
                "properties": {
                    "department": {"type": "string", "description": "Department to transfer to"},
                    "reason": {"type": "string", "description": "Reason for transfer"},
                },
                "required": ["department"],
            },
        ),
        Tool(
            name="end_call",
            description="End the call politely",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Reason for ending call"},
                },
            },
        ),
        Tool(
            name="book_appointment",
            description="Book an appointment for the caller",
            parameters={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Preferred date"},
                    "time": {"type": "string", "description": "Preferred time"},
                    "service": {"type": "string", "description": "Service requested"},
                },
                "required": ["date", "time"],
            },
        ),
    ]

    try:
        response = await llm_adapter.generate(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
        )

        # Process function calls
        action = None
        action_params = None

        for fc in response.function_calls:
            if fc.name == "transfer_call":
                action = "transfer"
                action_params = fc.arguments
            elif fc.name == "end_call":
                action = "hangup"
                action_params = fc.arguments
            elif fc.name == "book_appointment":
                action = "book_appointment"
                action_params = fc.arguments

        return VoiceTurnResponse(
            text=response.text,
            action=action,
            action_params=action_params,
        )

    except Exception as e:
        logger.error("Voice turn failed", error=str(e))
        # Return a safe fallback response
        return VoiceTurnResponse(
            text="I'm sorry, I encountered an issue. Could you please repeat that?",
        )


# ============================================
# RAG Endpoints
# ============================================

class IngestDocumentRequest(BaseModel):
    text: str
    agent_id: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[dict] = None


class IngestDocumentResponse(BaseModel):
    chunk_ids: list[str]
    chunk_count: int


class SearchRequest(BaseModel):
    query: str
    agent_id: Optional[str] = None
    top_k: int = 5
    filter_metadata: Optional[dict] = None


class SearchResultItem(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict = {}


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    context: str


@app.post("/rag/ingest", response_model=IngestDocumentResponse)
async def ingest_document(request: IngestDocumentRequest):
    """
    Ingest a document into the knowledge base.

    Chunks the document and adds it to the vector store.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        # Use agent-specific pipeline if agent_id provided
        if request.agent_id:
            if request.agent_id not in agent_rag_pipelines:
                agent_rag_pipelines[request.agent_id] = AgentRAGPipeline(request.agent_id)

            pipeline = agent_rag_pipelines[request.agent_id]
        else:
            pipeline = rag_pipeline

        chunk_ids = await pipeline.ingest_document(
            text=request.text,
            metadata=request.metadata,
            source=request.source,
        )

        return IngestDocumentResponse(
            chunk_ids=chunk_ids,
            chunk_count=len(chunk_ids),
        )

    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/search", response_model=SearchResponse)
async def search_knowledge(request: SearchRequest):
    """
    Search the knowledge base.

    Returns relevant documents and combined context.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        # Use agent-specific pipeline if agent_id provided
        if request.agent_id:
            if request.agent_id not in agent_rag_pipelines:
                agent_rag_pipelines[request.agent_id] = AgentRAGPipeline(request.agent_id)

            pipeline = agent_rag_pipelines[request.agent_id]
        else:
            pipeline = rag_pipeline

        result = await pipeline.retrieve(
            query=request.query,
            filter_metadata=request.filter_metadata,
        )

        return SearchResponse(
            query=result.query,
            results=[
                SearchResultItem(
                    id=r.id,
                    score=r.score,
                    text=r.text,
                    metadata=r.metadata,
                )
                for r in result.search_results.results
            ],
            context=result.context,
        )

    except Exception as e:
        logger.error("Search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class RAGGenerateRequest(BaseModel):
    query: str
    agent_id: Optional[str] = None
    conversation_history: list[MessageInput] = []
    system_prompt: Optional[str] = None
    use_rag: bool = True


class RAGGenerateResponse(BaseModel):
    text: str
    sources: list[dict] = []
    context_used: bool = False


@app.post("/rag/generate", response_model=RAGGenerateResponse)
async def generate_with_rag(request: RAGGenerateRequest):
    """
    Generate a response using RAG.

    Retrieves relevant context, then generates a response.
    """
    if not llm_adapter:
        raise HTTPException(status_code=503, detail="LLM adapter not initialized")

    try:
        sources = []
        augmented_query = request.query

        # Get RAG context if enabled
        if request.use_rag and rag_pipeline:
            if request.agent_id:
                if request.agent_id not in agent_rag_pipelines:
                    agent_rag_pipelines[request.agent_id] = AgentRAGPipeline(request.agent_id)
                pipeline = agent_rag_pipelines[request.agent_id]
            else:
                pipeline = rag_pipeline

            rag_result = await pipeline.retrieve(request.query)

            if rag_result.has_context:
                augmented_query = f"Context:\n{rag_result.context}\n\nQuestion: {request.query}"
                sources = rag_result.sources

        # Build messages
        messages = [
            Message(role=msg.role, content=msg.content)
            for msg in request.conversation_history
        ]
        messages.append(Message(role="user", content=augmented_query))

        # Generate response
        response = await llm_adapter.generate(
            messages=messages,
            system_prompt=request.system_prompt,
        )

        return RAGGenerateResponse(
            text=response.text,
            sources=sources,
            context_used=bool(sources),
        )

    except Exception as e:
        logger.error("RAG generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/rag/documents/{source}")
async def delete_documents(source: str, agent_id: Optional[str] = None):
    """Delete all documents from a source."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        if agent_id:
            if agent_id not in agent_rag_pipelines:
                return {"deleted": 0}
            pipeline = agent_rag_pipelines[agent_id]
        else:
            pipeline = rag_pipeline

        deleted = await pipeline.delete_by_source(source)
        return {"deleted": deleted, "source": source}

    except Exception as e:
        logger.error("Delete failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

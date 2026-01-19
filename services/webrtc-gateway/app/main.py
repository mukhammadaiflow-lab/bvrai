"""WebRTC Gateway - Main application entry point.

This service enables browser-based voice calls using WebRTC.
It handles WebSocket signaling, media processing, and bridges
to the Conversation Engine.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, WebSocket, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.sessions.manager import SessionManager
from app.signaling.handler import SignalingHandler

# Configure structured logging
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

# Global managers
session_manager = SessionManager()
signaling_handler = SignalingHandler(session_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(
        "Starting WebRTC Gateway",
        host=settings.host,
        port=settings.port,
    )

    # Start session manager
    await session_manager.start()

    yield

    # Cleanup
    logger.info("Shutting down WebRTC Gateway")
    await session_manager.stop()


# Create FastAPI application
app = FastAPI(
    title="WebRTC Gateway",
    description="""
    WebRTC Gateway for browser-based voice AI calls.

    ## Features

    - WebSocket signaling for WebRTC negotiation
    - TURN/STUN server configuration
    - Audio streaming to/from Conversation Engine
    - Real-time transcripts and agent responses

    ## WebSocket Protocol

    Connect to `/ws` with query parameter `agent_id` to start a session.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
class HealthResponse(BaseModel):
    status: str
    service: str
    active_sessions: int
    max_sessions: int


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="webrtc-gateway",
        active_sessions=session_manager.get_active_session_count(),
        max_sessions=settings.max_concurrent_sessions,
    )


@app.get("/", tags=["health"])
async def root():
    """Root endpoint."""
    return {
        "service": "WebRTC Gateway",
        "version": "1.0.0",
        "websocket": "/ws",
    }


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for WebRTC signaling.

    Clients connect here to establish WebRTC sessions.
    Protocol:
    1. Client connects and sends CONNECT message with agent_id
    2. Server responds with CONNECTED and ICE servers
    3. Client/Server exchange OFFER/ANSWER/ICE_CANDIDATE
    4. Client sends SESSION_START to begin conversation
    5. Server streams TRANSCRIPT and audio through WebRTC
    """
    await signaling_handler.handle_connection(websocket)


# REST endpoints for session management
class SessionInfo(BaseModel):
    session_id: str
    agent_id: str
    state: str
    created_at: str
    active: bool


@app.get("/sessions", response_model=list[SessionInfo], tags=["sessions"])
async def list_sessions():
    """List all active sessions."""
    sessions = session_manager.get_active_sessions()
    return [
        SessionInfo(
            session_id=s.session_id,
            agent_id=str(s.agent_id),
            state=s.state.value,
            created_at=s.created_at.isoformat(),
            active=s.conversation_active,
        )
        for s in sessions
    ]


@app.get("/sessions/{session_id}", response_model=SessionInfo, tags=["sessions"])
async def get_session(session_id: str):
    """Get session details."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    return SessionInfo(
        session_id=session.session_id,
        agent_id=str(session.agent_id),
        state=session.state.value,
        created_at=session.created_at.isoformat(),
        active=session.conversation_active,
    )


@app.delete("/sessions/{session_id}", tags=["sessions"])
async def end_session(session_id: str):
    """End a session."""
    success = await session_manager.end_session(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    return {"message": "Session ended", "session_id": session_id}


# ICE Server configuration endpoint
class IceServer(BaseModel):
    urls: str
    username: str = None
    credential: str = None


class IceConfig(BaseModel):
    ice_servers: list[IceServer]


@app.get("/ice-servers", response_model=IceConfig, tags=["configuration"])
async def get_ice_servers():
    """Get ICE server configuration.

    Returns STUN/TURN server configuration for WebRTC.
    """
    import time
    import hmac
    import hashlib
    import base64

    servers = []

    # Add STUN server
    if settings.stun_url:
        servers.append(IceServer(urls=settings.stun_url))

    # Add TURN server with time-limited credentials
    if settings.turn_enabled and settings.turn_url:
        username = f"{int(time.time()) + 86400}:webrtc"
        credential = base64.b64encode(
            hmac.new(
                settings.turn_secret.encode(),
                username.encode(),
                hashlib.sha1,
            ).digest()
        ).decode()

        servers.append(IceServer(
            urls=settings.turn_url,
            username=username,
            credential=credential,
        ))

    return IceConfig(ice_servers=servers)


# Stats endpoint
class GatewayStats(BaseModel):
    total_sessions: int
    active_sessions: int
    max_sessions: int


@app.get("/stats", response_model=GatewayStats, tags=["monitoring"])
async def get_stats():
    """Get gateway statistics."""
    return GatewayStats(
        total_sessions=session_manager.get_session_count(),
        active_sessions=session_manager.get_active_session_count(),
        max_sessions=settings.max_concurrent_sessions,
    )


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

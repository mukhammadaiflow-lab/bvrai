"""
Builder Engine Python SDK

A comprehensive Python SDK for the Builder Engine AI Voice Agent Platform.
Provides easy-to-use interfaces for managing agents, calls, conversations,
and all platform features.

Example:
    >>> from builderengine import BuilderEngine
    >>> client = BuilderEngine(api_key="your-api-key")
    >>> agent = client.agents.create(
    ...     name="Sales Agent",
    ...     voice_id="voice_abc123",
    ...     llm_config={"model": "gpt-4-turbo", "temperature": 0.7}
    ... )
    >>> call = client.calls.create(
    ...     agent_id=agent.id,
    ...     to_number="+1234567890"
    ... )
"""

__version__ = "1.0.0"
__author__ = "Builder Engine Team"
__license__ = "MIT"

from builderengine.client import BuilderEngine
from builderengine.models import (
    Agent,
    AgentConfig,
    Call,
    CallStatus,
    Conversation,
    Message,
    PhoneNumber,
    Voice,
    VoiceConfig,
    LLMConfig,
    Webhook,
    WebhookEvent,
    KnowledgeBase,
    Document,
    Workflow,
    WorkflowTrigger,
    Campaign,
    CampaignStatus,
    Analytics,
    Usage,
    Organization,
    User,
    APIKey,
)
from builderengine.exceptions import (
    BuilderEngineError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError,
    ConflictError,
    ServerError,
    WebSocketError,
    TimeoutError,
)
from builderengine.streaming import (
    StreamingClient,
    CallEventHandler,
    TranscriptionHandler,
)

__all__ = [
    # Main client
    "BuilderEngine",

    # Models
    "Agent",
    "AgentConfig",
    "Call",
    "CallStatus",
    "Conversation",
    "Message",
    "PhoneNumber",
    "Voice",
    "VoiceConfig",
    "LLMConfig",
    "Webhook",
    "WebhookEvent",
    "KnowledgeBase",
    "Document",
    "Workflow",
    "WorkflowTrigger",
    "Campaign",
    "CampaignStatus",
    "Analytics",
    "Usage",
    "Organization",
    "User",
    "APIKey",

    # Exceptions
    "BuilderEngineError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "NotFoundError",
    "ConflictError",
    "ServerError",
    "WebSocketError",
    "TimeoutError",

    # Streaming
    "StreamingClient",
    "CallEventHandler",
    "TranscriptionHandler",
]

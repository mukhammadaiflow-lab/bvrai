"""Builder Engine Voice AI SDK for Python."""

from bvrai_sdk.client import BvraiClient
from bvrai_sdk.agents import Agent, AgentConfig, AgentsAPI, Tool, ToolParameter
from bvrai_sdk.calls import Call, CallStatus, CallsAPI, CallDirection
from bvrai_sdk.knowledge import KnowledgeBase, KnowledgeAPI
from bvrai_sdk.analytics import (
    AnalyticsAPI,
    AnalyticsQueryBuilder,
    CallMetrics,
    AgentMetrics,
    UsageMetrics,
    RealtimeMetrics,
    TimeSeries,
    MetricType,
    AggregationType,
    TimeGranularity,
)
from bvrai_sdk.webhooks import (
    WebhooksAPI,
    Webhook,
    WebhookEvent,
    WebhookStatus,
    WebhookDelivery,
    WebhookHandler,
    WebhookSignatureVerifier,
    WebhookPayload,
    WebhookBuilder,
)
from bvrai_sdk.phone_numbers import (
    PhoneNumbersAPI,
    PhoneNumber,
    PhoneNumberType,
    PhoneNumberStatus,
    PhoneNumberCapability,
    PhoneNumberConfig,
    AvailableNumber,
    PhoneNumberSearchBuilder,
)
from bvrai_sdk.streaming import (
    StreamingConnection,
    StreamingSession,
    StreamEvent,
    StreamEventType,
    TranscriptEvent,
    AudioFrame,
    AudioStreamPlayer,
    AudioStreamRecorder,
)
from bvrai_sdk.exceptions import (
    BvraiError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ValidationError,
    ServerError,
    ConnectionError,
)

__version__ = "1.0.0"

__all__ = [
    # Client
    "BvraiClient",

    # Agents
    "Agent",
    "AgentConfig",
    "AgentsAPI",
    "Tool",
    "ToolParameter",

    # Calls
    "Call",
    "CallStatus",
    "CallsAPI",
    "CallDirection",

    # Knowledge
    "KnowledgeBase",
    "KnowledgeAPI",

    # Analytics
    "AnalyticsAPI",
    "AnalyticsQueryBuilder",
    "CallMetrics",
    "AgentMetrics",
    "UsageMetrics",
    "RealtimeMetrics",
    "TimeSeries",
    "MetricType",
    "AggregationType",
    "TimeGranularity",

    # Webhooks
    "WebhooksAPI",
    "Webhook",
    "WebhookEvent",
    "WebhookStatus",
    "WebhookDelivery",
    "WebhookHandler",
    "WebhookSignatureVerifier",
    "WebhookPayload",
    "WebhookBuilder",

    # Phone Numbers
    "PhoneNumbersAPI",
    "PhoneNumber",
    "PhoneNumberType",
    "PhoneNumberStatus",
    "PhoneNumberCapability",
    "PhoneNumberConfig",
    "AvailableNumber",
    "PhoneNumberSearchBuilder",

    # Streaming
    "StreamingConnection",
    "StreamingSession",
    "StreamEvent",
    "StreamEventType",
    "TranscriptEvent",
    "AudioFrame",
    "AudioStreamPlayer",
    "AudioStreamRecorder",

    # Exceptions
    "BvraiError",
    "AuthenticationError",
    "NotFoundError",
    "RateLimitError",
    "ValidationError",
    "ServerError",
    "ConnectionError",
]

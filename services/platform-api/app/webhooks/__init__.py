"""
Webhook System Module

Complete webhook infrastructure:
- Event subscription and dispatch
- Payload transformation
- Delivery tracking
- Retry handling
"""

from app.webhooks.routes import router
from app.webhooks.delivery import (
    WebhookDeliveryService,
    WebhookConfig,
    WebhookDelivery,
    DeliveryAttempt,
    DeliveryStatus,
    WebhookSigner,
    get_delivery_service,
    send_webhook,
    broadcast_event,
)

# Engine
from app.webhooks.engine import (
    WebhookEngine,
    WebhookEndpoint,
    WebhookDelivery as EngineDelivery,
    WebhookEvent,
    WebhookStatus,
    WebhookEventType,
    SignatureMethod,
    SignatureGenerator,
    WebhookDispatcher,
    get_webhook_engine,
    create_event,
    create_endpoint,
)

# Handlers
from app.webhooks.handlers import (
    EventHandler,
    PayloadTransformer,
    PayloadFilter,
    EventBatcher,
    EventRouter,
    WebhookMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    EventEnricher,
    EventProcessor,
    CallEventHandler,
    TranscriptEventHandler,
    create_event_processor,
    create_metrics_middleware,
)

__all__ = [
    # Routes
    "router",
    # Delivery service
    "WebhookDeliveryService",
    "WebhookConfig",
    "WebhookDelivery",
    "DeliveryAttempt",
    "DeliveryStatus",
    "WebhookSigner",
    "get_delivery_service",
    "send_webhook",
    "broadcast_event",
    # Engine
    "WebhookEngine",
    "WebhookEndpoint",
    "EngineDelivery",
    "WebhookEvent",
    "WebhookStatus",
    "WebhookEventType",
    "SignatureMethod",
    "SignatureGenerator",
    "WebhookDispatcher",
    "get_webhook_engine",
    "create_event",
    "create_endpoint",
    # Handlers
    "EventHandler",
    "PayloadTransformer",
    "PayloadFilter",
    "EventBatcher",
    "EventRouter",
    "WebhookMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "EventEnricher",
    "EventProcessor",
    "CallEventHandler",
    "TranscriptEventHandler",
    "create_event_processor",
    "create_metrics_middleware",
]

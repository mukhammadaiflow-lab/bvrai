"""
Developer Platform Module

This module provides comprehensive developer platform features for voice agent
applications, including API key management, rate limiting, usage tracking,
webhook management, and SDK support.

Features:
- API Key Management: Secure key generation, rotation, revocation
- Rate Limiting: Multiple strategies (fixed window, sliding window, token bucket)
- Usage Tracking: API call tracking, quota management, analytics
- Webhook Management: Event delivery with retry and signature verification
- SDK Code Generation: Multi-language code snippets

Example usage:

    from bvrai_core.developer import (
        DeveloperPlatformService,
        APIKeyType,
        RateLimitStrategy,
        WebhookEventType,
        SDKLanguage,
    )

    # Initialize the service
    service = DeveloperPlatformService()

    # Set up organization with starter plan
    setup = await service.setup_organization(
        organization_id="org_123",
        plan_name="professional",
    )
    print(f"API Key: {setup['full_key']}")  # Store this securely!

    # Create additional API key
    api_key, full_key = await service.api_keys.create_key(
        organization_id="org_123",
        name="Production Key",
        key_type=APIKeyType.LIVE,
        description="Main production API key",
    )

    # Authenticate a request
    allowed, key, error = await service.authenticate_request(
        api_key_value="sk_live_xxx",
        resource="calls",
        action="create",
        ip_address="192.168.1.1",
        endpoint="/v1/calls",
    )

    if not allowed:
        print(f"Request denied: {error}")

    # Record API usage
    await service.record_request(
        api_key=key,
        endpoint="/v1/calls",
        method="POST",
        status_code=201,
        response_time_ms=150.5,
    )

    # Create webhook endpoint
    webhook = await service.webhooks.create_endpoint(
        organization_id="org_123",
        url="https://example.com/webhooks",
        events=[
            WebhookEventType.CALL_STARTED,
            WebhookEventType.CALL_ENDED,
            WebhookEventType.TRANSCRIPTION_READY,
        ],
    )

    # Dispatch webhook event
    await service.dispatch_webhook_event(
        organization_id="org_123",
        event_type=WebhookEventType.CALL_ENDED,
        payload={
            "call_id": "call_xxx",
            "duration": 120,
            "outcome": "completed",
        },
    )

    # Generate SDK code snippet
    python_snippet = service.sdk_generator.generate_auth_snippet(
        SDKLanguage.PYTHON,
        api_key="sk_live_xxx",
    )
    print(python_snippet)

    # Get developer dashboard
    dashboard = await service.get_developer_dashboard("org_123")
    print(f"API Calls Used: {dashboard['usage']['total_requests']}")

    # Upgrade plan
    await service.upgrade_plan("org_123", "enterprise")
"""

# Base types and enums
from .base import (
    # Enums
    APIKeyType,
    APIKeyStatus,
    RateLimitScope,
    RateLimitStrategy,
    QuotaPeriod,
    WebhookEventType,
    SDKLanguage,
    # API Key types
    APIKeyScope,
    APIKey,
    # Rate limiting types
    RateLimitConfig,
    RateLimitState,
    RateLimitResult,
    # Usage types
    UsageQuota,
    APIUsageEvent,
    # Webhook types
    WebhookEndpoint,
    # Exceptions
    DeveloperPlatformError,
    APIKeyError,
    RateLimitError,
    QuotaExceededError,
    WebhookError,
)

# Services
from .service import (
    # API Key Manager
    APIKeyManager,
    # Rate Limiters
    RateLimiter,
    FixedWindowRateLimiter,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    RateLimitManager,
    # Usage Tracker
    UsageTracker,
    # Webhook Manager
    WebhookDelivery,
    WebhookManager,
    # SDK Generator
    SDKCodeGenerator,
    # Main Service
    DeveloperPlatformService,
)


__all__ = [
    # Enums
    "APIKeyType",
    "APIKeyStatus",
    "RateLimitScope",
    "RateLimitStrategy",
    "QuotaPeriod",
    "WebhookEventType",
    "SDKLanguage",
    # API Key types
    "APIKeyScope",
    "APIKey",
    # Rate limiting types
    "RateLimitConfig",
    "RateLimitState",
    "RateLimitResult",
    # Usage types
    "UsageQuota",
    "APIUsageEvent",
    # Webhook types
    "WebhookEndpoint",
    # API Key Manager
    "APIKeyManager",
    # Rate Limiters
    "RateLimiter",
    "FixedWindowRateLimiter",
    "SlidingWindowRateLimiter",
    "TokenBucketRateLimiter",
    "RateLimitManager",
    # Usage Tracker
    "UsageTracker",
    # Webhook Manager
    "WebhookDelivery",
    "WebhookManager",
    # SDK Generator
    "SDKCodeGenerator",
    # Main Service
    "DeveloperPlatformService",
    # Exceptions
    "DeveloperPlatformError",
    "APIKeyError",
    "RateLimitError",
    "QuotaExceededError",
    "WebhookError",
]

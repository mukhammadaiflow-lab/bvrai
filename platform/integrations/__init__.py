"""
Integration Hub Package

This package provides a comprehensive integration system for the voice agent platform,
enabling connections to external services including CRM systems, calendars,
cloud storage, and more.

Architecture:
- base.py: Core types, interfaces, and abstract provider classes
- registry.py: Provider registration and instance management
- service.py: High-level service layer for integration operations
- crm/: CRM provider implementations (Salesforce, HubSpot)
- calendar/: Calendar provider implementations (Google, Outlook)
- storage/: Storage provider implementations (S3)

Example usage:

    from platform.integrations import (
        IntegrationService,
        IntegrationServiceConfig,
        create_integration_service,
    )

    # Create service
    service = create_integration_service()

    # Start OAuth flow for Salesforce
    auth_url = await service.start_oauth_flow(
        organization_id="org_123",
        provider_name="salesforce",
        redirect_uri="https://app.example.com/oauth/callback",
    )

    # After OAuth callback, complete the flow
    integration = await service.complete_oauth_flow(
        state=state_from_callback,
        code=authorization_code,
    )

    # Use the integration
    contacts, cursor = await service.registry.get_crm_provider(
        integration.id
    ).list_contacts(limit=100)

    # Log a call
    await service.log_call_to_crm(
        integration_id=integration.id,
        organization_id="org_123",
        call_log=CallLogEntry(...),
    )

    # Schedule a meeting
    event = await service.schedule_meeting(
        integration_id=calendar_integration.id,
        organization_id="org_123",
        title="Sales Call",
        start_time=datetime.utcnow() + timedelta(hours=2),
        end_time=datetime.utcnow() + timedelta(hours=3),
        attendee_emails=["client@example.com"],
    )
"""

# Base types and interfaces
from .base import (
    # Enums
    IntegrationType,
    IntegrationStatus,
    AuthType,
    SyncDirection,
    SyncStatus,
    # Credential types
    OAuthCredentials,
    APICredentials,
    # Configuration
    IntegrationConfig,
    SyncRecord,
    # Data models
    ExternalContact,
    ExternalDeal,
    CalendarEvent,
    CallLogEntry,
    Calendar,
    StorageObject,
    SyncState,
    # Provider base classes
    IntegrationProvider,
    FlexibleIntegrationProvider,
    CRMProvider,
    CalendarProvider,
    StorageProvider,
    MessagingProvider,
    # Errors
    IntegrationError,
    AuthenticationError,
    RateLimitError,
    SyncError,
    ProviderNotFoundError,
)

# Registry
from .registry import (
    ProviderCategory,
    ProviderDefinition,
    IntegrationInstance,
    CredentialStore,
    InMemoryCredentialStore,
    IntegrationRegistry,
    get_default_provider_definitions,
    create_registry_with_defaults,
)

# Service
from .service import (
    OAuthState,
    OAuthSession,
    SyncJob,
    IntegrationServiceConfig,
    IntegrationService,
    create_integration_service,
)

# CRM Providers
from .crm import (
    SalesforceProvider,
    HubSpotProvider,
    get_crm_provider,
    list_crm_providers,
    CRM_PROVIDERS,
)

# Calendar Providers
from .calendar import (
    GoogleCalendarProvider,
    OutlookCalendarProvider,
    get_calendar_provider,
    list_calendar_providers,
    CALENDAR_PROVIDERS,
)

# Storage Providers
from .storage import (
    S3Provider,
    get_storage_provider,
    list_storage_providers,
    STORAGE_PROVIDERS,
)


__all__ = [
    # === Base Types ===
    # Enums
    "IntegrationType",
    "IntegrationStatus",
    "AuthType",
    "SyncDirection",
    "SyncStatus",
    # Credentials
    "OAuthCredentials",
    "APICredentials",
    # Configuration
    "IntegrationConfig",
    "SyncRecord",
    # Data Models
    "ExternalContact",
    "ExternalDeal",
    "CalendarEvent",
    "CallLogEntry",
    "Calendar",
    "StorageObject",
    "SyncState",
    # Provider Base Classes
    "IntegrationProvider",
    "FlexibleIntegrationProvider",
    "CRMProvider",
    "CalendarProvider",
    "StorageProvider",
    "MessagingProvider",
    # Errors
    "IntegrationError",
    "AuthenticationError",
    "RateLimitError",
    "SyncError",
    "ProviderNotFoundError",
    # === Registry ===
    "ProviderCategory",
    "ProviderDefinition",
    "IntegrationInstance",
    "CredentialStore",
    "InMemoryCredentialStore",
    "IntegrationRegistry",
    "get_default_provider_definitions",
    "create_registry_with_defaults",
    # === Service ===
    "OAuthState",
    "OAuthSession",
    "SyncJob",
    "IntegrationServiceConfig",
    "IntegrationService",
    "create_integration_service",
    # === CRM Providers ===
    "SalesforceProvider",
    "HubSpotProvider",
    "get_crm_provider",
    "list_crm_providers",
    "CRM_PROVIDERS",
    # === Calendar Providers ===
    "GoogleCalendarProvider",
    "OutlookCalendarProvider",
    "get_calendar_provider",
    "list_calendar_providers",
    "CALENDAR_PROVIDERS",
    # === Storage Providers ===
    "S3Provider",
    "get_storage_provider",
    "list_storage_providers",
    "STORAGE_PROVIDERS",
]

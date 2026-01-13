"""
Integration Registry

This module provides a centralized registry for managing integration providers,
enabling discovery, instantiation, and lifecycle management of all integration types.

Features:
- Provider registration and discovery
- Integration instance management
- Credential storage abstraction
- Health monitoring
- Event dispatching
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar

from .base import (
    IntegrationProvider,
    IntegrationType,
    AuthType,
    IntegrationStatus,
    IntegrationError,
    CRMProvider,
    CalendarProvider,
    StorageProvider,
    MessagingProvider,
)


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=IntegrationProvider)


class ProviderCategory(str, Enum):
    """Provider category classification."""
    CRM = "crm"
    CALENDAR = "calendar"
    STORAGE = "storage"
    MESSAGING = "messaging"
    EMAIL = "email"
    TELEPHONY = "telephony"
    ANALYTICS = "analytics"
    PAYMENT = "payment"
    CUSTOM = "custom"


@dataclass
class ProviderDefinition:
    """Definition of an integration provider."""
    name: str
    display_name: str
    category: ProviderCategory
    provider_class: Type[IntegrationProvider]
    auth_type: AuthType
    description: str = ""
    icon_url: str = ""
    documentation_url: str = ""
    oauth_config: Optional[Dict[str, Any]] = None
    required_credentials: List[str] = field(default_factory=list)
    optional_settings: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    rate_limits: Optional[Dict[str, int]] = None
    beta: bool = False
    deprecated: bool = False


@dataclass
class IntegrationInstance:
    """Runtime instance of an integration."""
    id: str
    organization_id: str
    provider_name: str
    display_name: str
    provider: IntegrationProvider
    status: IntegrationStatus = IntegrationStatus.DISCONNECTED
    connected_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "provider_name": self.provider_name,
            "display_name": self.display_name,
            "status": self.status.value,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_status": self.health_status,
            "error_count": self.error_count,
            "created_at": self.created_at.isoformat(),
        }


class CredentialStore:
    """
    Abstract credential storage interface.

    Implementations should securely store and retrieve integration credentials.
    """

    async def store_credentials(
        self,
        integration_id: str,
        organization_id: str,
        credentials: Dict[str, Any],
    ) -> None:
        """Store credentials for an integration."""
        raise NotImplementedError

    async def get_credentials(
        self,
        integration_id: str,
        organization_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve credentials for an integration."""
        raise NotImplementedError

    async def update_credentials(
        self,
        integration_id: str,
        organization_id: str,
        updates: Dict[str, Any],
    ) -> None:
        """Update stored credentials."""
        raise NotImplementedError

    async def delete_credentials(
        self,
        integration_id: str,
        organization_id: str,
    ) -> None:
        """Delete credentials for an integration."""
        raise NotImplementedError


class InMemoryCredentialStore(CredentialStore):
    """In-memory credential store for development/testing."""

    def __init__(self):
        self._credentials: Dict[str, Dict[str, Any]] = {}

    def _get_key(self, integration_id: str, organization_id: str) -> str:
        return f"{organization_id}:{integration_id}"

    async def store_credentials(
        self,
        integration_id: str,
        organization_id: str,
        credentials: Dict[str, Any],
    ) -> None:
        key = self._get_key(integration_id, organization_id)
        self._credentials[key] = credentials.copy()

    async def get_credentials(
        self,
        integration_id: str,
        organization_id: str,
    ) -> Optional[Dict[str, Any]]:
        key = self._get_key(integration_id, organization_id)
        creds = self._credentials.get(key)
        return creds.copy() if creds else None

    async def update_credentials(
        self,
        integration_id: str,
        organization_id: str,
        updates: Dict[str, Any],
    ) -> None:
        key = self._get_key(integration_id, organization_id)
        if key in self._credentials:
            self._credentials[key].update(updates)

    async def delete_credentials(
        self,
        integration_id: str,
        organization_id: str,
    ) -> None:
        key = self._get_key(integration_id, organization_id)
        self._credentials.pop(key, None)


class IntegrationRegistry:
    """
    Central registry for integration providers.

    Manages provider definitions, creates integration instances,
    and handles lifecycle events.

    Example usage:

        registry = IntegrationRegistry()

        # Register providers
        registry.register_provider(ProviderDefinition(
            name="salesforce",
            display_name="Salesforce",
            category=ProviderCategory.CRM,
            provider_class=SalesforceProvider,
            auth_type=AuthType.OAUTH2,
        ))

        # List available providers
        providers = registry.list_providers()

        # Create integration instance
        integration = await registry.create_integration(
            provider_name="salesforce",
            integration_id="int_123",
            organization_id="org_456",
            credentials={...},
            settings={...},
        )

        # Connect
        await registry.connect(integration.id)
    """

    def __init__(
        self,
        credential_store: Optional[CredentialStore] = None,
    ):
        """
        Initialize the integration registry.

        Args:
            credential_store: Storage backend for credentials
        """
        self._providers: Dict[str, ProviderDefinition] = {}
        self._instances: Dict[str, IntegrationInstance] = {}
        self._credential_store = credential_store or InMemoryCredentialStore()

        # Event handlers
        self._on_connected: List[Callable] = []
        self._on_disconnected: List[Callable] = []
        self._on_error: List[Callable] = []
        self._on_credentials_updated: List[Callable] = []

        # Health check configuration
        self._health_check_interval = 300  # 5 minutes
        self._health_check_task: Optional[asyncio.Task] = None

    # =========================================================================
    # Provider Registration
    # =========================================================================

    def register_provider(self, definition: ProviderDefinition) -> None:
        """
        Register an integration provider.

        Args:
            definition: Provider definition
        """
        if definition.deprecated:
            logger.warning(f"Registering deprecated provider: {definition.name}")

        self._providers[definition.name] = definition
        logger.info(f"Registered integration provider: {definition.name}")

    def unregister_provider(self, name: str) -> bool:
        """
        Unregister an integration provider.

        Args:
            name: Provider name

        Returns:
            True if provider was unregistered
        """
        if name in self._providers:
            del self._providers[name]
            logger.info(f"Unregistered integration provider: {name}")
            return True
        return False

    def get_provider(self, name: str) -> Optional[ProviderDefinition]:
        """
        Get a provider definition by name.

        Args:
            name: Provider name

        Returns:
            Provider definition or None
        """
        return self._providers.get(name)

    def list_providers(
        self,
        category: Optional[ProviderCategory] = None,
        include_beta: bool = True,
        include_deprecated: bool = False,
    ) -> List[ProviderDefinition]:
        """
        List registered providers.

        Args:
            category: Filter by category
            include_beta: Include beta providers
            include_deprecated: Include deprecated providers

        Returns:
            List of provider definitions
        """
        providers = []
        for provider in self._providers.values():
            if category and provider.category != category:
                continue
            if not include_beta and provider.beta:
                continue
            if not include_deprecated and provider.deprecated:
                continue
            providers.append(provider)

        return sorted(providers, key=lambda p: p.display_name)

    def list_providers_by_type(
        self,
        integration_type: IntegrationType,
    ) -> List[ProviderDefinition]:
        """
        List providers by integration type.

        Args:
            integration_type: Type of integration

        Returns:
            List of provider definitions
        """
        type_to_category = {
            IntegrationType.CRM: ProviderCategory.CRM,
            IntegrationType.CALENDAR: ProviderCategory.CALENDAR,
            IntegrationType.STORAGE: ProviderCategory.STORAGE,
            IntegrationType.EMAIL: ProviderCategory.EMAIL,
            IntegrationType.MESSAGING: ProviderCategory.MESSAGING,
        }

        category = type_to_category.get(integration_type)
        if category:
            return self.list_providers(category=category)
        return []

    # =========================================================================
    # Integration Instance Management
    # =========================================================================

    async def create_integration(
        self,
        provider_name: str,
        integration_id: str,
        organization_id: str,
        credentials: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None,
        display_name: Optional[str] = None,
        auto_connect: bool = False,
    ) -> IntegrationInstance:
        """
        Create a new integration instance.

        Args:
            provider_name: Name of the provider
            integration_id: Unique integration identifier
            organization_id: Organization identifier
            credentials: Provider credentials
            settings: Provider settings
            display_name: Human-readable name
            auto_connect: Automatically connect after creation

        Returns:
            Created integration instance

        Raises:
            IntegrationError: If provider not found or creation fails
        """
        provider_def = self._providers.get(provider_name)
        if not provider_def:
            raise IntegrationError(f"Unknown provider: {provider_name}")

        # Store credentials securely
        await self._credential_store.store_credentials(
            integration_id,
            organization_id,
            credentials,
        )

        # Create provider instance
        provider = provider_def.provider_class(
            integration_id=integration_id,
            organization_id=organization_id,
            credentials=credentials,
            settings=settings or {},
        )

        # Set up token refresh callback
        provider.set_token_refresh_callback(self._handle_token_refresh)

        # Create integration instance
        instance = IntegrationInstance(
            id=integration_id,
            organization_id=organization_id,
            provider_name=provider_name,
            display_name=display_name or provider_def.display_name,
            provider=provider,
        )

        self._instances[integration_id] = instance
        logger.info(
            f"Created integration {integration_id} for provider {provider_name}"
        )

        if auto_connect:
            await self.connect(integration_id)

        return instance

    async def get_integration(
        self,
        integration_id: str,
        organization_id: Optional[str] = None,
    ) -> Optional[IntegrationInstance]:
        """
        Get an integration instance by ID.

        Args:
            integration_id: Integration identifier
            organization_id: Organization identifier for validation

        Returns:
            Integration instance or None
        """
        instance = self._instances.get(integration_id)

        if instance and organization_id:
            if instance.organization_id != organization_id:
                return None

        return instance

    async def list_integrations(
        self,
        organization_id: str,
        provider_name: Optional[str] = None,
        category: Optional[ProviderCategory] = None,
        status: Optional[IntegrationStatus] = None,
    ) -> List[IntegrationInstance]:
        """
        List integration instances for an organization.

        Args:
            organization_id: Organization identifier
            provider_name: Filter by provider
            category: Filter by category
            status: Filter by status

        Returns:
            List of integration instances
        """
        instances = []

        for instance in self._instances.values():
            if instance.organization_id != organization_id:
                continue
            if provider_name and instance.provider_name != provider_name:
                continue
            if status and instance.status != status:
                continue
            if category:
                provider_def = self._providers.get(instance.provider_name)
                if not provider_def or provider_def.category != category:
                    continue

            instances.append(instance)

        return instances

    async def delete_integration(
        self,
        integration_id: str,
        organization_id: Optional[str] = None,
    ) -> bool:
        """
        Delete an integration instance.

        Args:
            integration_id: Integration identifier
            organization_id: Organization identifier for validation

        Returns:
            True if deleted
        """
        instance = await self.get_integration(integration_id, organization_id)
        if not instance:
            return False

        # Disconnect if connected
        if instance.status == IntegrationStatus.CONNECTED:
            await self.disconnect(integration_id)

        # Delete credentials
        await self._credential_store.delete_credentials(
            integration_id,
            instance.organization_id,
        )

        # Remove instance
        del self._instances[integration_id]
        logger.info(f"Deleted integration {integration_id}")

        return True

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self, integration_id: str) -> bool:
        """
        Connect an integration.

        Args:
            integration_id: Integration identifier

        Returns:
            True if connection successful
        """
        instance = self._instances.get(integration_id)
        if not instance:
            raise IntegrationError(f"Integration not found: {integration_id}")

        try:
            success = await instance.provider.connect()

            if success:
                instance.status = IntegrationStatus.CONNECTED
                instance.connected_at = datetime.utcnow()
                instance.error_count = 0
                instance.health_status = "healthy"

                # Emit connected event
                for handler in self._on_connected:
                    try:
                        await handler(instance)
                    except Exception as e:
                        logger.error(f"Error in connected handler: {e}")

            else:
                instance.status = IntegrationStatus.AUTH_FAILED
                instance.health_status = "unhealthy"
                instance.error_count += 1

            return success

        except Exception as e:
            instance.status = IntegrationStatus.ERROR
            instance.health_status = "unhealthy"
            instance.error_count += 1

            # Emit error event
            for handler in self._on_error:
                try:
                    await handler(instance, e)
                except Exception as he:
                    logger.error(f"Error in error handler: {he}")

            raise IntegrationError(f"Connection failed: {e}")

    async def disconnect(self, integration_id: str) -> bool:
        """
        Disconnect an integration.

        Args:
            integration_id: Integration identifier

        Returns:
            True if disconnection successful
        """
        instance = self._instances.get(integration_id)
        if not instance:
            return False

        try:
            success = await instance.provider.disconnect()

            if success:
                instance.status = IntegrationStatus.DISCONNECTED
                instance.connected_at = None

                # Emit disconnected event
                for handler in self._on_disconnected:
                    try:
                        await handler(instance)
                    except Exception as e:
                        logger.error(f"Error in disconnected handler: {e}")

            return success

        except Exception as e:
            logger.error(f"Error disconnecting integration {integration_id}: {e}")
            return False

    async def reconnect(self, integration_id: str) -> bool:
        """
        Reconnect an integration.

        Args:
            integration_id: Integration identifier

        Returns:
            True if reconnection successful
        """
        instance = self._instances.get(integration_id)
        if not instance:
            return False

        await self.disconnect(integration_id)
        return await self.connect(integration_id)

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    async def health_check(self, integration_id: str) -> Dict[str, Any]:
        """
        Perform health check on an integration.

        Args:
            integration_id: Integration identifier

        Returns:
            Health check results
        """
        instance = self._instances.get(integration_id)
        if not instance:
            return {"status": "not_found", "error": "Integration not found"}

        try:
            health = await instance.provider.health_check()

            instance.last_health_check = datetime.utcnow()
            instance.health_status = health.get("status", "unknown")

            if health.get("status") == "healthy":
                instance.error_count = 0
            elif health.get("status") == "unhealthy":
                instance.error_count += 1

            return health

        except Exception as e:
            instance.error_count += 1
            instance.health_status = "error"
            return {
                "status": "error",
                "error": str(e),
            }

    async def health_check_all(
        self,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform health checks on all integrations.

        Args:
            organization_id: Filter by organization

        Returns:
            Dictionary mapping integration IDs to health results
        """
        results = {}

        instances = list(self._instances.values())
        if organization_id:
            instances = [i for i in instances if i.organization_id == organization_id]

        # Run health checks concurrently
        tasks = [
            self.health_check(instance.id)
            for instance in instances
        ]

        if tasks:
            health_results = await asyncio.gather(*tasks, return_exceptions=True)

            for instance, result in zip(instances, health_results):
                if isinstance(result, Exception):
                    results[instance.id] = {
                        "status": "error",
                        "error": str(result),
                    }
                else:
                    results[instance.id] = result

        return results

    async def start_health_monitoring(
        self,
        interval_seconds: int = 300,
    ) -> None:
        """
        Start periodic health monitoring.

        Args:
            interval_seconds: Interval between health checks
        """
        self._health_check_interval = interval_seconds

        if self._health_check_task:
            self._health_check_task.cancel()

        self._health_check_task = asyncio.create_task(
            self._health_monitoring_loop()
        )
        logger.info(f"Started health monitoring with {interval_seconds}s interval")

    async def stop_health_monitoring(self) -> None:
        """Stop periodic health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
            logger.info("Stopped health monitoring")

    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self.health_check_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_connected(self, handler: Callable) -> None:
        """Register handler for connection events."""
        self._on_connected.append(handler)

    def on_disconnected(self, handler: Callable) -> None:
        """Register handler for disconnection events."""
        self._on_disconnected.append(handler)

    def on_error(self, handler: Callable) -> None:
        """Register handler for error events."""
        self._on_error.append(handler)

    def on_credentials_updated(self, handler: Callable) -> None:
        """Register handler for credential update events."""
        self._on_credentials_updated.append(handler)

    async def _handle_token_refresh(
        self,
        integration_id: str,
        new_credentials: Dict[str, Any],
    ) -> None:
        """Handle token refresh callback from providers."""
        instance = self._instances.get(integration_id)
        if not instance:
            return

        # Update stored credentials
        await self._credential_store.update_credentials(
            integration_id,
            instance.organization_id,
            new_credentials,
        )

        logger.info(f"Updated credentials for integration {integration_id}")

        # Emit credentials updated event
        for handler in self._on_credentials_updated:
            try:
                await handler(instance, new_credentials)
            except Exception as e:
                logger.error(f"Error in credentials updated handler: {e}")

    # =========================================================================
    # Typed Provider Access
    # =========================================================================

    def get_crm_provider(self, integration_id: str) -> Optional[CRMProvider]:
        """Get integration as CRM provider."""
        instance = self._instances.get(integration_id)
        if instance and isinstance(instance.provider, CRMProvider):
            return instance.provider
        return None

    def get_calendar_provider(self, integration_id: str) -> Optional[CalendarProvider]:
        """Get integration as calendar provider."""
        instance = self._instances.get(integration_id)
        if instance and isinstance(instance.provider, CalendarProvider):
            return instance.provider
        return None

    def get_storage_provider(self, integration_id: str) -> Optional[StorageProvider]:
        """Get integration as storage provider."""
        instance = self._instances.get(integration_id)
        if instance and isinstance(instance.provider, StorageProvider):
            return instance.provider
        return None

    def get_messaging_provider(self, integration_id: str) -> Optional[MessagingProvider]:
        """Get integration as messaging provider."""
        instance = self._instances.get(integration_id)
        if instance and isinstance(instance.provider, MessagingProvider):
            return instance.provider
        return None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(
        self,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get registry statistics.

        Args:
            organization_id: Filter by organization

        Returns:
            Statistics dictionary
        """
        instances = list(self._instances.values())
        if organization_id:
            instances = [i for i in instances if i.organization_id == organization_id]

        status_counts = {}
        category_counts = {}
        health_counts = {}

        for instance in instances:
            # Status
            status = instance.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            # Category
            provider_def = self._providers.get(instance.provider_name)
            if provider_def:
                category = provider_def.category.value
                category_counts[category] = category_counts.get(category, 0) + 1

            # Health
            health = instance.health_status
            health_counts[health] = health_counts.get(health, 0) + 1

        return {
            "total_providers": len(self._providers),
            "total_integrations": len(instances),
            "by_status": status_counts,
            "by_category": category_counts,
            "by_health": health_counts,
        }


# =========================================================================
# Default Provider Definitions
# =========================================================================

def get_default_provider_definitions() -> List[ProviderDefinition]:
    """
    Get default provider definitions for common integrations.

    Returns:
        List of provider definitions
    """
    from .crm.salesforce import SalesforceProvider
    from .crm.hubspot import HubSpotProvider
    from .calendar.google import GoogleCalendarProvider
    from .calendar.outlook import OutlookCalendarProvider
    from .storage.s3 import S3Provider

    return [
        # CRM Providers
        ProviderDefinition(
            name="salesforce",
            display_name="Salesforce",
            category=ProviderCategory.CRM,
            provider_class=SalesforceProvider,
            auth_type=AuthType.OAUTH2,
            description="Enterprise CRM platform for sales, service, and marketing",
            required_credentials=["access_token", "refresh_token", "instance_url"],
            optional_settings=["api_version", "sandbox"],
            capabilities=[
                "list_contacts", "get_contact", "create_contact", "update_contact",
                "search_contacts", "find_by_phone", "log_call",
                "list_deals", "get_deal", "create_deal", "update_deal",
            ],
        ),
        ProviderDefinition(
            name="hubspot",
            display_name="HubSpot",
            category=ProviderCategory.CRM,
            provider_class=HubSpotProvider,
            auth_type=AuthType.OAUTH2,
            description="All-in-one marketing, sales, and service platform",
            required_credentials=["access_token", "refresh_token"],
            optional_settings=["portal_id"],
            capabilities=[
                "list_contacts", "get_contact", "create_contact", "update_contact",
                "search_contacts", "find_by_phone", "log_call",
                "list_deals", "get_deal", "create_deal", "update_deal",
            ],
        ),
        # Calendar Providers
        ProviderDefinition(
            name="google_calendar",
            display_name="Google Calendar",
            category=ProviderCategory.CALENDAR,
            provider_class=GoogleCalendarProvider,
            auth_type=AuthType.OAUTH2,
            description="Google Calendar for scheduling and availability",
            required_credentials=["access_token", "refresh_token", "client_id", "client_secret"],
            optional_settings=["default_calendar_id", "time_zone", "auto_add_meet"],
            capabilities=[
                "list_calendars", "list_events", "create_event", "update_event",
                "delete_event", "get_availability", "schedule_meeting",
                "find_meeting_time", "sync_events",
            ],
        ),
        ProviderDefinition(
            name="outlook_calendar",
            display_name="Microsoft Outlook Calendar",
            category=ProviderCategory.CALENDAR,
            provider_class=OutlookCalendarProvider,
            auth_type=AuthType.OAUTH2,
            description="Microsoft 365 Calendar with Teams integration",
            required_credentials=["access_token", "refresh_token", "client_id", "client_secret", "tenant_id"],
            optional_settings=["default_calendar_id", "time_zone", "auto_add_teams"],
            capabilities=[
                "list_calendars", "list_events", "create_event", "update_event",
                "delete_event", "get_availability", "schedule_meeting",
                "find_meeting_time", "sync_events",
            ],
        ),
        # Storage Providers
        ProviderDefinition(
            name="s3",
            display_name="Amazon S3",
            category=ProviderCategory.STORAGE,
            provider_class=S3Provider,
            auth_type=AuthType.API_KEY,
            description="AWS Simple Storage Service for file storage",
            required_credentials=["access_key_id", "secret_access_key"],
            optional_settings=["bucket", "region", "encryption", "storage_class"],
            capabilities=[
                "list_objects", "get_object", "upload_object", "download_object",
                "delete_object", "copy_object", "generate_presigned_url",
                "multipart_upload", "list_buckets",
            ],
        ),
    ]


def create_registry_with_defaults(
    credential_store: Optional[CredentialStore] = None,
) -> IntegrationRegistry:
    """
    Create an integration registry with default providers registered.

    Args:
        credential_store: Optional credential store

    Returns:
        Configured IntegrationRegistry
    """
    registry = IntegrationRegistry(credential_store=credential_store)

    for definition in get_default_provider_definitions():
        registry.register_provider(definition)

    return registry

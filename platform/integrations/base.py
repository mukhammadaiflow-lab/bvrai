"""
Integration Hub Base Types

Core types, interfaces, and base classes for the integration system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union


class IntegrationType(str, Enum):
    """Types of integrations."""
    CRM = "crm"
    CALENDAR = "calendar"
    EMAIL = "email"
    SMS = "sms"
    STORAGE = "storage"
    ANALYTICS = "analytics"
    PAYMENT = "payment"
    CUSTOM = "custom"


class IntegrationStatus(str, Enum):
    """Integration connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    EXPIRED = "expired"
    REVOKED = "revoked"


class AuthType(str, Enum):
    """Authentication types for integrations."""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    BEARER = "bearer"
    CUSTOM = "custom"


class SyncDirection(str, Enum):
    """Data sync direction."""
    INBOUND = "inbound"  # From external service to platform
    OUTBOUND = "outbound"  # From platform to external service
    BIDIRECTIONAL = "bidirectional"


class SyncStatus(str, Enum):
    """Sync operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class OAuthCredentials:
    """OAuth2 credentials."""

    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    scope: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() >= self.expires_at - timedelta(minutes=5)


@dataclass
class APICredentials:
    """API key credentials."""

    api_key: str
    api_secret: Optional[str] = None
    additional_params: Dict[str, str] = field(default_factory=dict)


@dataclass
class IntegrationConfig:
    """Configuration for an integration."""

    id: str
    organization_id: str
    integration_type: IntegrationType
    provider: str
    name: str

    # Authentication
    auth_type: AuthType
    credentials: Optional[Union[OAuthCredentials, APICredentials, Dict[str, Any]]] = None

    # Status
    status: IntegrationStatus = IntegrationStatus.DISCONNECTED
    last_connected_at: Optional[datetime] = None
    last_sync_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Settings
    sync_enabled: bool = True
    sync_interval_minutes: int = 60
    sync_direction: SyncDirection = SyncDirection.BIDIRECTIONAL

    # Field mappings
    field_mappings: Dict[str, str] = field(default_factory=dict)

    # Webhook configuration
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "integration_type": self.integration_type.value,
            "provider": self.provider,
            "name": self.name,
            "auth_type": self.auth_type.value,
            "status": self.status.value,
            "last_connected_at": self.last_connected_at.isoformat() if self.last_connected_at else None,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "sync_enabled": self.sync_enabled,
            "sync_interval_minutes": self.sync_interval_minutes,
            "sync_direction": self.sync_direction.value,
            "field_mappings": self.field_mappings,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SyncRecord:
    """Record of a sync operation."""

    id: str
    integration_id: str
    organization_id: str
    direction: SyncDirection
    status: SyncStatus

    # Statistics
    records_processed: int = 0
    records_created: int = 0
    records_updated: int = 0
    records_failed: int = 0

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None

    # Pagination state
    last_cursor: Optional[str] = None
    next_cursor: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "integration_id": self.integration_id,
            "organization_id": self.organization_id,
            "direction": self.direction.value,
            "status": self.status.value,
            "records_processed": self.records_processed,
            "records_created": self.records_created,
            "records_updated": self.records_updated,
            "records_failed": self.records_failed,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }


@dataclass
class ExternalContact:
    """Contact from an external CRM."""

    external_id: str
    provider: str

    # Basic info
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    mobile: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None

    # Address
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None

    # CRM-specific
    owner_id: Optional[str] = None
    lead_source: Optional[str] = None
    lifecycle_stage: Optional[str] = None

    # Custom fields
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None

    # Internal mapping
    internal_contact_id: Optional[str] = None

    @property
    def full_name(self) -> str:
        """Get full name."""
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p)

    @property
    def primary_phone(self) -> Optional[str]:
        """Get primary phone number."""
        return self.phone or self.mobile


@dataclass
class ExternalDeal:
    """Deal/Opportunity from an external CRM."""

    external_id: str
    provider: str
    name: str

    # Deal info
    amount: Optional[float] = None
    currency: str = "USD"
    stage: Optional[str] = None
    probability: Optional[float] = None

    # Associations
    contact_id: Optional[str] = None
    company_id: Optional[str] = None
    owner_id: Optional[str] = None

    # Dates
    close_date: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Status
    is_won: bool = False
    is_closed: bool = False

    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalendarEvent:
    """Calendar event."""

    external_id: str
    provider: str

    # Event info
    title: str
    description: Optional[str] = None
    location: Optional[str] = None

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime = field(default_factory=datetime.utcnow)
    all_day: bool = False
    timezone: str = "UTC"

    # Attendees
    organizer_email: Optional[str] = None
    attendees: List[Dict[str, Any]] = field(default_factory=list)

    # Conference
    conference_url: Optional[str] = None
    conference_type: Optional[str] = None

    # Recurrence
    recurring: bool = False
    recurrence_rule: Optional[str] = None

    # Status
    status: str = "confirmed"  # confirmed, tentative, cancelled

    # Reminders
    reminders: List[Dict[str, Any]] = field(default_factory=list)

    # Internal mapping
    internal_call_id: Optional[str] = None
    internal_contact_id: Optional[str] = None

    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallLogEntry:
    """Call log entry for CRM sync."""

    call_id: str
    organization_id: str

    # Call details
    direction: str  # inbound, outbound
    from_number: str
    to_number: str
    duration_seconds: int
    status: str  # completed, missed, voicemail, failed

    # Timestamps
    started_at: datetime
    ended_at: Optional[datetime] = None

    # Agent info
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None

    # Content
    recording_url: Optional[str] = None
    transcript_summary: Optional[str] = None
    sentiment: Optional[str] = None

    # Outcome
    outcome: Optional[str] = None
    disposition: Optional[str] = None
    follow_up_date: Optional[datetime] = None

    # Associations
    contact_id: Optional[str] = None
    deal_id: Optional[str] = None

    # Sync status
    synced_to_crm: bool = False
    external_log_id: Optional[str] = None

    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Generic type for data records
T = TypeVar('T')


class IntegrationProvider(ABC, Generic[T]):
    """
    Abstract base class for integration providers.

    All integration providers (CRM, Calendar, etc.) must implement this interface.
    """

    # Provider metadata
    PROVIDER_NAME: str = ""
    PROVIDER_TYPE: IntegrationType = IntegrationType.CUSTOM
    AUTH_TYPE: AuthType = AuthType.OAUTH2
    SCOPES: List[str] = []

    def __init__(self, config: IntegrationConfig):
        """Initialize provider with configuration."""
        self.config = config
        self._client = None

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the external service.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the external service."""
        pass

    @abstractmethod
    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """
        Test the connection.

        Returns:
            Tuple of (success, error_message)
        """
        pass

    @abstractmethod
    async def refresh_credentials(self) -> bool:
        """
        Refresh OAuth credentials if needed.

        Returns:
            True if refresh successful or not needed
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        return self.config.status == IntegrationStatus.CONNECTED


class CRMProvider(IntegrationProvider[ExternalContact]):
    """Base class for CRM providers."""

    PROVIDER_TYPE = IntegrationType.CRM

    @abstractmethod
    async def list_contacts(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        updated_since: Optional[datetime] = None,
    ) -> Tuple[List[ExternalContact], Optional[str]]:
        """
        List contacts from CRM.

        Returns:
            Tuple of (contacts, next_cursor)
        """
        pass

    @abstractmethod
    async def get_contact(self, external_id: str) -> Optional[ExternalContact]:
        """Get a contact by external ID."""
        pass

    @abstractmethod
    async def create_contact(self, contact: ExternalContact) -> ExternalContact:
        """Create a contact in CRM."""
        pass

    @abstractmethod
    async def update_contact(
        self,
        external_id: str,
        updates: Dict[str, Any],
    ) -> ExternalContact:
        """Update a contact in CRM."""
        pass

    @abstractmethod
    async def search_contacts(
        self,
        query: str,
        limit: int = 20,
    ) -> List[ExternalContact]:
        """Search contacts in CRM."""
        pass

    @abstractmethod
    async def log_call(self, entry: CallLogEntry) -> str:
        """
        Log a call to the CRM.

        Returns:
            External ID of the created activity/log
        """
        pass

    async def list_deals(
        self,
        contact_id: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[ExternalDeal], Optional[str]]:
        """List deals from CRM."""
        return ([], None)

    async def create_deal(self, deal: ExternalDeal) -> ExternalDeal:
        """Create a deal in CRM."""
        raise NotImplementedError("Deal creation not supported")


class CalendarProvider(IntegrationProvider[CalendarEvent]):
    """Base class for calendar providers."""

    PROVIDER_TYPE = IntegrationType.CALENDAR

    @abstractmethod
    async def list_calendars(self) -> List[Dict[str, Any]]:
        """List available calendars."""
        pass

    @abstractmethod
    async def list_events(
        self,
        calendar_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> List[CalendarEvent]:
        """List events from a calendar."""
        pass

    @abstractmethod
    async def get_event(
        self,
        calendar_id: str,
        event_id: str,
    ) -> Optional[CalendarEvent]:
        """Get a specific event."""
        pass

    @abstractmethod
    async def create_event(
        self,
        calendar_id: str,
        event: CalendarEvent,
    ) -> CalendarEvent:
        """Create a calendar event."""
        pass

    @abstractmethod
    async def update_event(
        self,
        calendar_id: str,
        event_id: str,
        updates: Dict[str, Any],
    ) -> CalendarEvent:
        """Update a calendar event."""
        pass

    @abstractmethod
    async def delete_event(
        self,
        calendar_id: str,
        event_id: str,
    ) -> bool:
        """Delete a calendar event."""
        pass

    @abstractmethod
    async def get_availability(
        self,
        calendar_id: str,
        start_time: datetime,
        end_time: datetime,
        slot_duration_minutes: int = 30,
    ) -> List[Dict[str, datetime]]:
        """
        Get available time slots.

        Returns:
            List of {"start": datetime, "end": datetime}
        """
        pass


class MessagingProvider(IntegrationProvider[Dict[str, Any]]):
    """Base class for messaging providers (SMS, Email)."""

    @abstractmethod
    async def send_message(
        self,
        to: str,
        content: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send a message."""
        pass

    @abstractmethod
    async def get_message_status(
        self,
        message_id: str,
    ) -> Dict[str, Any]:
        """Get status of a sent message."""
        pass


class StorageProvider(IntegrationProvider[bytes]):
    """Base class for storage providers."""

    PROVIDER_TYPE = IntegrationType.STORAGE

    @abstractmethod
    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload data to storage.

        Returns:
            URL or path to uploaded file
        """
        pass

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Download data from storage."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data from storage."""
        pass

    @abstractmethod
    async def get_signed_url(
        self,
        key: str,
        expires_in_seconds: int = 3600,
    ) -> str:
        """Get a signed URL for temporary access."""
        pass

    @abstractmethod
    async def list_objects(
        self,
        prefix: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List objects with a given prefix."""
        pass


# Helper type for provider lookup
from typing import Tuple


# Errors

class IntegrationError(Exception):
    """Base integration error."""

    def __init__(self, message: str, code: str = "integration_error"):
        self.message = message
        self.code = code
        super().__init__(message)


class AuthenticationError(IntegrationError):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "authentication_error")


class RateLimitError(IntegrationError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        message = "Rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after} seconds"
        super().__init__(message, "rate_limit_error")


class SyncError(IntegrationError):
    """Sync operation failed."""

    def __init__(self, message: str, failed_records: int = 0):
        self.failed_records = failed_records
        super().__init__(message, "sync_error")


class ProviderNotFoundError(IntegrationError):
    """Provider not found."""

    def __init__(self, provider: str):
        super().__init__(f"Provider not found: {provider}", "provider_not_found")


# =========================================================================
# Extended Types for Provider Implementations
# =========================================================================

# Add AUTH_FAILED status
IntegrationStatus.AUTH_FAILED = "auth_failed"


@dataclass
class Calendar:
    """Calendar object for calendar integrations."""
    id: str
    name: str
    description: str = ""
    time_zone: str = "UTC"
    color: Optional[str] = None
    is_primary: bool = False
    access_role: str = "reader"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageObject:
    """Object stored in cloud storage."""
    key: str
    bucket: str
    size: int = 0
    content_type: str = "application/octet-stream"
    last_modified: Optional[datetime] = None
    etag: Optional[str] = None
    storage_class: Optional[str] = None
    version_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncState:
    """State for incremental synchronization."""
    resource_type: str
    cursor: Optional[str] = None
    last_sync: Optional[datetime] = None
    full_sync_needed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================================
# Flexible Provider Base Classes
# =========================================================================

class FlexibleIntegrationProvider(ABC, Generic[T]):
    """
    Flexible base class for integration providers with direct credential passing.

    This is an alternative to IntegrationProvider that accepts credentials directly
    rather than through an IntegrationConfig object.
    """

    PROVIDER_NAME: str = ""
    AUTH_TYPE: AuthType = AuthType.OAUTH2

    def __init__(
        self,
        integration_id: str,
        organization_id: str,
        credentials: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize provider.

        Args:
            integration_id: Unique integration identifier
            organization_id: Organization this integration belongs to
            credentials: Provider-specific credentials
            settings: Optional provider settings
        """
        self.integration_id = integration_id
        self.organization_id = organization_id
        self.credentials = credentials
        self.settings = settings or {}

        self._status = IntegrationStatus.DISCONNECTED
        self._on_token_refresh: Optional[Callable] = None

    @property
    def status(self) -> IntegrationStatus:
        """Get current status."""
        return self._status

    def set_token_refresh_callback(self, callback: Callable) -> None:
        """Set callback for token refresh events."""
        self._on_token_refresh = callback

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the external service."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the external service."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the integration."""
        pass


# Update provider base classes to support flexible initialization
# These are aliased to support both old and new patterns

class CRMProvider(FlexibleIntegrationProvider[ExternalContact]):
    """
    Base class for CRM providers.

    Provides contact management, deal tracking, and call logging
    capabilities for external CRM systems.
    """

    @abstractmethod
    async def list_contacts(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        updated_since: Optional[datetime] = None,
    ) -> Tuple[List[ExternalContact], Optional[str]]:
        """List contacts from CRM."""
        pass

    @abstractmethod
    async def get_contact(self, contact_id: str) -> Optional[ExternalContact]:
        """Get a contact by ID."""
        pass

    @abstractmethod
    async def create_contact(self, contact: ExternalContact) -> ExternalContact:
        """Create a contact in CRM."""
        pass

    @abstractmethod
    async def update_contact(
        self,
        contact_id: str,
        updates: Dict[str, Any],
    ) -> ExternalContact:
        """Update a contact in CRM."""
        pass

    @abstractmethod
    async def search_contacts(
        self,
        query: str,
        limit: int = 20,
    ) -> List[ExternalContact]:
        """Search contacts in CRM."""
        pass

    @abstractmethod
    async def find_contact_by_phone(self, phone: str) -> Optional[ExternalContact]:
        """Find a contact by phone number."""
        pass

    @abstractmethod
    async def log_call(self, entry: CallLogEntry) -> str:
        """Log a call to the CRM."""
        pass

    async def list_deals(
        self,
        contact_id: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[ExternalDeal], Optional[str]]:
        """List deals from CRM."""
        return ([], None)

    async def get_deal(self, deal_id: str) -> Optional[ExternalDeal]:
        """Get a deal by ID."""
        return None

    async def create_deal(self, deal: ExternalDeal) -> ExternalDeal:
        """Create a deal in CRM."""
        raise NotImplementedError("Deal creation not supported")

    async def update_deal(
        self,
        deal_id: str,
        updates: Dict[str, Any],
    ) -> ExternalDeal:
        """Update a deal in CRM."""
        raise NotImplementedError("Deal update not supported")


# Redefine CalendarEvent for flexible providers
@dataclass
class CalendarEvent:
    """Calendar event for flexible calendar providers."""
    id: str
    calendar_id: Optional[str]
    title: str
    start_time: datetime
    end_time: datetime
    description: Optional[str] = None
    location: Optional[str] = None
    time_zone: Optional[str] = None
    all_day: bool = False
    attendees: List[Dict[str, Any]] = field(default_factory=list)
    organizer: Optional[Dict[str, Any]] = None
    reminders: List[Dict[str, Any]] = field(default_factory=list)
    recurrence: Optional[List[str]] = None
    meeting_link: Optional[str] = None
    status: Optional[str] = None
    visibility: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CalendarProvider(FlexibleIntegrationProvider[CalendarEvent]):
    """
    Base class for calendar providers.

    Provides calendar management, event scheduling, and availability
    checking capabilities for external calendar systems.
    """

    @abstractmethod
    async def list_calendars(self) -> List[Calendar]:
        """List available calendars."""
        pass

    @abstractmethod
    async def get_calendar(self, calendar_id: str) -> Optional[Calendar]:
        """Get a specific calendar."""
        pass

    @abstractmethod
    async def list_events(
        self,
        calendar_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> List[CalendarEvent]:
        """List events from a calendar."""
        pass

    @abstractmethod
    async def get_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
    ) -> Optional[CalendarEvent]:
        """Get a specific event."""
        pass

    @abstractmethod
    async def create_event(self, event: CalendarEvent) -> CalendarEvent:
        """Create a calendar event."""
        pass

    @abstractmethod
    async def update_event(self, event: CalendarEvent) -> CalendarEvent:
        """Update a calendar event."""
        pass

    @abstractmethod
    async def delete_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
        send_updates: str = "all",
    ) -> bool:
        """Delete a calendar event."""
        pass

    @abstractmethod
    async def get_availability(
        self,
        calendar_ids: List[str],
        start_time: datetime,
        end_time: datetime,
        slot_duration_minutes: int = 30,
    ) -> List[Dict[str, datetime]]:
        """Get available time slots."""
        pass

    async def get_free_busy(
        self,
        calendar_ids: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, List[Dict[str, datetime]]]:
        """Get free/busy information."""
        return {}

    async def sync_events(
        self,
        calendar_id: Optional[str] = None,
        full_sync: bool = False,
    ) -> Tuple[List[CalendarEvent], List[str], Optional[str]]:
        """Sync events incrementally."""
        return ([], [], None)

    async def get_sync_state(self, resource_type: str) -> Optional[SyncState]:
        """Get sync state for a resource type."""
        return None


class StorageProvider(FlexibleIntegrationProvider[StorageObject]):
    """
    Base class for storage providers.

    Provides file storage, retrieval, and management capabilities
    for cloud storage systems.
    """

    @abstractmethod
    async def list_objects(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
        limit: int = 1000,
        cursor: Optional[str] = None,
    ) -> Tuple[List[StorageObject], Optional[str]]:
        """List objects in storage."""
        pass

    @abstractmethod
    async def get_object(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> Optional[StorageObject]:
        """Get object metadata."""
        pass

    @abstractmethod
    async def upload_object(
        self,
        key: str,
        data: bytes,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        storage_class: Optional[str] = None,
    ) -> StorageObject:
        """Upload an object."""
        pass

    @abstractmethod
    async def download_object(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        """Download object content."""
        pass

    @abstractmethod
    async def delete_object(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Delete an object."""
        pass

    @abstractmethod
    def generate_presigned_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in: int = 3600,
        method: str = "GET",
        content_type: Optional[str] = None,
    ) -> str:
        """Generate a presigned URL."""
        pass


class MessagingProvider(FlexibleIntegrationProvider[Dict[str, Any]]):
    """
    Base class for messaging providers.

    Provides message sending capabilities for SMS, email, and
    other messaging systems.
    """

    @abstractmethod
    async def send_message(
        self,
        to: str,
        content: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send a message."""
        pass

    @abstractmethod
    async def get_message_status(
        self,
        message_id: str,
    ) -> Dict[str, Any]:
        """Get status of a sent message."""
        pass

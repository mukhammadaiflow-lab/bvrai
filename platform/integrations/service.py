"""
Integration Service

This module provides the main service layer for managing integrations,
coordinating between the registry, providers, and external systems.

Features:
- OAuth flow management
- Integration CRUD operations
- Sync orchestration
- Webhook handling
- Usage analytics
"""

import asyncio
import hashlib
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .base import (
    IntegrationType,
    IntegrationStatus,
    IntegrationError,
    AuthType,
    ExternalContact,
    CalendarEvent,
    CallLogEntry,
    SyncState,
)
from .registry import (
    IntegrationRegistry,
    IntegrationInstance,
    ProviderDefinition,
    ProviderCategory,
    CredentialStore,
    InMemoryCredentialStore,
    create_registry_with_defaults,
)


logger = logging.getLogger(__name__)


class OAuthState(str, Enum):
    """OAuth flow states."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class OAuthSession:
    """OAuth authorization session."""
    id: str
    organization_id: str
    provider_name: str
    state: str
    redirect_uri: str
    status: OAuthState = OAuthState.PENDING
    code_verifier: Optional[str] = None  # For PKCE
    scopes: List[str] = field(default_factory=list)
    integration_id: Optional[str] = None  # Set after completion
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=10))


@dataclass
class SyncJob:
    """Synchronization job."""
    id: str
    integration_id: str
    organization_id: str
    resource_type: str  # contacts, events, etc.
    status: str = "pending"
    progress: float = 0.0
    total_items: int = 0
    synced_items: int = 0
    failed_items: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "integration_id": self.integration_id,
            "organization_id": self.organization_id,
            "resource_type": self.resource_type,
            "status": self.status,
            "progress": self.progress,
            "total_items": self.total_items,
            "synced_items": self.synced_items,
            "failed_items": self.failed_items,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class IntegrationServiceConfig:
    """Integration service configuration."""
    oauth_redirect_base_url: str = "http://localhost:8000"
    sync_batch_size: int = 100
    sync_concurrent_limit: int = 5
    webhook_secret_key: str = ""
    enable_sync_scheduling: bool = True
    default_sync_interval_hours: int = 1
    max_sync_retries: int = 3


class IntegrationService:
    """
    Main integration service for the voice agent platform.

    Provides high-level API for managing integrations including:
    - OAuth flow handling
    - Integration lifecycle management
    - Data synchronization
    - Webhook processing
    - Usage tracking

    Example usage:

        service = IntegrationService()

        # Start OAuth flow
        auth_url = await service.start_oauth_flow(
            organization_id="org_123",
            provider_name="salesforce",
            redirect_uri="https://app.example.com/oauth/callback",
        )

        # Complete OAuth flow (in callback handler)
        integration = await service.complete_oauth_flow(
            state="abc123",
            code="authorization_code",
        )

        # Sync contacts
        job = await service.sync_contacts(integration.id)

        # Log a call
        await service.log_call_to_crm(
            integration_id=integration.id,
            call_log=CallLogEntry(...),
        )
    """

    def __init__(
        self,
        config: Optional[IntegrationServiceConfig] = None,
        registry: Optional[IntegrationRegistry] = None,
        credential_store: Optional[CredentialStore] = None,
    ):
        """
        Initialize the integration service.

        Args:
            config: Service configuration
            registry: Integration registry (created with defaults if not provided)
            credential_store: Credential storage backend
        """
        self.config = config or IntegrationServiceConfig()
        self._credential_store = credential_store or InMemoryCredentialStore()
        self._registry = registry or create_registry_with_defaults(self._credential_store)

        # OAuth sessions
        self._oauth_sessions: Dict[str, OAuthSession] = {}

        # Sync jobs
        self._sync_jobs: Dict[str, SyncJob] = {}
        self._sync_states: Dict[str, SyncState] = {}

        # Usage tracking
        self._usage_stats: Dict[str, Dict[str, int]] = {}

        # Event handlers
        self._on_sync_complete: List[Callable] = []
        self._on_contact_synced: List[Callable] = []
        self._on_call_logged: List[Callable] = []

        # Background tasks
        self._sync_scheduler_task: Optional[asyncio.Task] = None

        # Register event handlers on registry
        self._registry.on_connected(self._handle_integration_connected)
        self._registry.on_disconnected(self._handle_integration_disconnected)
        self._registry.on_error(self._handle_integration_error)

    @property
    def registry(self) -> IntegrationRegistry:
        """Get the integration registry."""
        return self._registry

    # =========================================================================
    # OAuth Flow Management
    # =========================================================================

    async def start_oauth_flow(
        self,
        organization_id: str,
        provider_name: str,
        redirect_uri: str,
        scopes: Optional[List[str]] = None,
    ) -> str:
        """
        Start OAuth authorization flow.

        Args:
            organization_id: Organization initiating the flow
            provider_name: Integration provider name
            redirect_uri: URI to redirect after authorization
            scopes: Optional OAuth scopes

        Returns:
            Authorization URL to redirect user to

        Raises:
            IntegrationError: If provider doesn't support OAuth
        """
        provider_def = self._registry.get_provider(provider_name)
        if not provider_def:
            raise IntegrationError(f"Unknown provider: {provider_name}")

        if provider_def.auth_type != AuthType.OAUTH2:
            raise IntegrationError(f"Provider {provider_name} doesn't use OAuth2")

        # Generate state and session
        state = secrets.token_urlsafe(32)
        session_id = f"oauth_{secrets.token_hex(16)}"

        session = OAuthSession(
            id=session_id,
            organization_id=organization_id,
            provider_name=provider_name,
            state=state,
            redirect_uri=redirect_uri,
            scopes=scopes or [],
        )

        self._oauth_sessions[state] = session

        # Build authorization URL based on provider
        provider_class = provider_def.provider_class
        oauth_config = provider_def.oauth_config or {}

        # Most OAuth providers have a class method for generating auth URL
        if hasattr(provider_class, "get_authorization_url"):
            auth_url = provider_class.get_authorization_url(
                client_id=oauth_config.get("client_id", ""),
                redirect_uri=redirect_uri,
                state=state,
                scopes=scopes,
            )
        else:
            raise IntegrationError(
                f"Provider {provider_name} doesn't implement OAuth authorization"
            )

        logger.info(
            f"Started OAuth flow for {provider_name} "
            f"(org: {organization_id}, session: {session_id})"
        )

        return auth_url

    async def complete_oauth_flow(
        self,
        state: str,
        code: str,
        settings: Optional[Dict[str, Any]] = None,
    ) -> IntegrationInstance:
        """
        Complete OAuth authorization flow.

        Args:
            state: State parameter from callback
            code: Authorization code from callback
            settings: Optional integration settings

        Returns:
            Created integration instance

        Raises:
            IntegrationError: If flow fails or state is invalid
        """
        session = self._oauth_sessions.get(state)
        if not session:
            raise IntegrationError("Invalid or expired OAuth state")

        if session.status != OAuthState.PENDING:
            raise IntegrationError(f"OAuth flow already {session.status.value}")

        if datetime.utcnow() > session.expires_at:
            session.status = OAuthState.EXPIRED
            raise IntegrationError("OAuth session expired")

        provider_def = self._registry.get_provider(session.provider_name)
        if not provider_def:
            raise IntegrationError(f"Unknown provider: {session.provider_name}")

        try:
            # Exchange code for tokens
            provider_class = provider_def.provider_class
            oauth_config = provider_def.oauth_config or {}

            if hasattr(provider_class, "exchange_code"):
                tokens = await provider_class.exchange_code(
                    code=code,
                    client_id=oauth_config.get("client_id", ""),
                    client_secret=oauth_config.get("client_secret", ""),
                    redirect_uri=session.redirect_uri,
                )
            else:
                raise IntegrationError(
                    f"Provider {session.provider_name} doesn't implement token exchange"
                )

            # Merge OAuth config into credentials
            credentials = {
                **tokens,
                "client_id": oauth_config.get("client_id"),
                "client_secret": oauth_config.get("client_secret"),
            }

            # Create integration
            integration_id = f"int_{secrets.token_hex(12)}"

            integration = await self._registry.create_integration(
                provider_name=session.provider_name,
                integration_id=integration_id,
                organization_id=session.organization_id,
                credentials=credentials,
                settings=settings,
                auto_connect=True,
            )

            # Update session
            session.status = OAuthState.COMPLETED
            session.integration_id = integration_id

            logger.info(
                f"Completed OAuth flow for {session.provider_name} "
                f"(org: {session.organization_id}, integration: {integration_id})"
            )

            return integration

        except Exception as e:
            session.status = OAuthState.FAILED
            session.error = str(e)
            raise IntegrationError(f"OAuth flow failed: {e}")

    def get_oauth_session(self, state: str) -> Optional[OAuthSession]:
        """Get OAuth session by state."""
        return self._oauth_sessions.get(state)

    async def cleanup_expired_oauth_sessions(self) -> int:
        """Remove expired OAuth sessions."""
        now = datetime.utcnow()
        expired = [
            state for state, session in self._oauth_sessions.items()
            if now > session.expires_at
        ]

        for state in expired:
            del self._oauth_sessions[state]

        return len(expired)

    # =========================================================================
    # Integration Management
    # =========================================================================

    async def create_integration(
        self,
        organization_id: str,
        provider_name: str,
        credentials: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None,
        display_name: Optional[str] = None,
    ) -> IntegrationInstance:
        """
        Create a new integration (for non-OAuth providers).

        Args:
            organization_id: Organization identifier
            provider_name: Provider name
            credentials: Provider credentials
            settings: Provider settings
            display_name: Human-readable name

        Returns:
            Created integration instance
        """
        integration_id = f"int_{secrets.token_hex(12)}"

        integration = await self._registry.create_integration(
            provider_name=provider_name,
            integration_id=integration_id,
            organization_id=organization_id,
            credentials=credentials,
            settings=settings,
            display_name=display_name,
            auto_connect=True,
        )

        logger.info(
            f"Created integration {integration_id} for {provider_name} "
            f"(org: {organization_id})"
        )

        return integration

    async def get_integration(
        self,
        integration_id: str,
        organization_id: str,
    ) -> Optional[IntegrationInstance]:
        """Get an integration by ID."""
        return await self._registry.get_integration(integration_id, organization_id)

    async def list_integrations(
        self,
        organization_id: str,
        category: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[IntegrationInstance]:
        """List integrations for an organization."""
        provider_category = None
        if category:
            try:
                provider_category = ProviderCategory(category)
            except ValueError:
                pass

        integration_status = None
        if status:
            try:
                integration_status = IntegrationStatus(status)
            except ValueError:
                pass

        return await self._registry.list_integrations(
            organization_id=organization_id,
            category=provider_category,
            status=integration_status,
        )

    async def update_integration(
        self,
        integration_id: str,
        organization_id: str,
        settings: Optional[Dict[str, Any]] = None,
        display_name: Optional[str] = None,
    ) -> Optional[IntegrationInstance]:
        """Update integration settings."""
        instance = await self._registry.get_integration(integration_id, organization_id)
        if not instance:
            return None

        if settings:
            instance.provider.settings.update(settings)
        if display_name:
            instance.display_name = display_name

        return instance

    async def delete_integration(
        self,
        integration_id: str,
        organization_id: str,
    ) -> bool:
        """Delete an integration."""
        return await self._registry.delete_integration(integration_id, organization_id)

    async def reconnect_integration(
        self,
        integration_id: str,
        organization_id: str,
    ) -> bool:
        """Reconnect an integration."""
        instance = await self._registry.get_integration(integration_id, organization_id)
        if not instance:
            return False

        return await self._registry.reconnect(integration_id)

    # =========================================================================
    # CRM Operations
    # =========================================================================

    async def sync_contacts(
        self,
        integration_id: str,
        organization_id: str,
        full_sync: bool = False,
    ) -> SyncJob:
        """
        Start contact synchronization from CRM.

        Args:
            integration_id: CRM integration ID
            organization_id: Organization ID
            full_sync: Force full sync instead of incremental

        Returns:
            Sync job
        """
        instance = await self._registry.get_integration(integration_id, organization_id)
        if not instance:
            raise IntegrationError(f"Integration not found: {integration_id}")

        crm = self._registry.get_crm_provider(integration_id)
        if not crm:
            raise IntegrationError(f"Integration {integration_id} is not a CRM")

        # Create sync job
        job_id = f"sync_{secrets.token_hex(12)}"
        job = SyncJob(
            id=job_id,
            integration_id=integration_id,
            organization_id=organization_id,
            resource_type="contacts",
            status="running",
            started_at=datetime.utcnow(),
        )
        self._sync_jobs[job_id] = job

        # Run sync in background
        asyncio.create_task(
            self._run_contact_sync(job, crm, full_sync)
        )

        return job

    async def _run_contact_sync(
        self,
        job: SyncJob,
        crm,
        full_sync: bool,
    ) -> None:
        """Run contact synchronization."""
        try:
            cursor = None
            synced = 0
            failed = 0

            while True:
                contacts, next_cursor = await crm.list_contacts(
                    limit=self.config.sync_batch_size,
                    cursor=cursor,
                )

                job.total_items += len(contacts)

                for contact in contacts:
                    try:
                        # Emit contact synced event
                        for handler in self._on_contact_synced:
                            await handler(job.organization_id, contact)
                        synced += 1
                    except Exception as e:
                        failed += 1
                        job.errors.append({
                            "contact_id": contact.id,
                            "error": str(e),
                        })

                job.synced_items = synced
                job.failed_items = failed
                job.progress = (synced + failed) / max(job.total_items, 1)

                cursor = next_cursor
                if not cursor:
                    break

            job.status = "completed"
            job.completed_at = datetime.utcnow()

            # Emit sync complete event
            for handler in self._on_sync_complete:
                await handler(job)

        except Exception as e:
            job.status = "failed"
            job.errors.append({"error": str(e)})
            logger.error(f"Contact sync failed for job {job.id}: {e}")

    async def log_call_to_crm(
        self,
        integration_id: str,
        organization_id: str,
        call_log: CallLogEntry,
    ) -> Optional[str]:
        """
        Log a call to CRM.

        Args:
            integration_id: CRM integration ID
            organization_id: Organization ID
            call_log: Call log entry

        Returns:
            CRM activity/task ID or None
        """
        instance = await self._registry.get_integration(integration_id, organization_id)
        if not instance:
            raise IntegrationError(f"Integration not found: {integration_id}")

        crm = self._registry.get_crm_provider(integration_id)
        if not crm:
            raise IntegrationError(f"Integration {integration_id} is not a CRM")

        try:
            activity_id = await crm.log_call(call_log)

            # Track usage
            self._track_usage(organization_id, "crm_call_logs", 1)

            # Emit call logged event
            for handler in self._on_call_logged:
                await handler(organization_id, integration_id, call_log, activity_id)

            logger.info(
                f"Logged call to CRM {integration_id} "
                f"(contact: {call_log.contact_id}, activity: {activity_id})"
            )

            return activity_id

        except Exception as e:
            logger.error(f"Failed to log call to CRM {integration_id}: {e}")
            raise IntegrationError(f"Failed to log call: {e}")

    async def find_contact_by_phone(
        self,
        integration_id: str,
        organization_id: str,
        phone: str,
    ) -> Optional[ExternalContact]:
        """
        Find a contact by phone number in CRM.

        Args:
            integration_id: CRM integration ID
            organization_id: Organization ID
            phone: Phone number to search

        Returns:
            Contact or None
        """
        crm = self._registry.get_crm_provider(integration_id)
        if not crm:
            raise IntegrationError(f"Integration {integration_id} is not a CRM")

        contact = await crm.find_contact_by_phone(phone)

        # Track usage
        self._track_usage(organization_id, "crm_lookups", 1)

        return contact

    async def search_contacts(
        self,
        integration_id: str,
        organization_id: str,
        query: str,
        limit: int = 10,
    ) -> List[ExternalContact]:
        """
        Search contacts in CRM.

        Args:
            integration_id: CRM integration ID
            organization_id: Organization ID
            query: Search query
            limit: Maximum results

        Returns:
            List of matching contacts
        """
        crm = self._registry.get_crm_provider(integration_id)
        if not crm:
            raise IntegrationError(f"Integration {integration_id} is not a CRM")

        contacts = await crm.search_contacts(query, limit)

        # Track usage
        self._track_usage(organization_id, "crm_searches", 1)

        return contacts

    # =========================================================================
    # Calendar Operations
    # =========================================================================

    async def check_availability(
        self,
        integration_id: str,
        organization_id: str,
        calendar_ids: List[str],
        start_time: datetime,
        end_time: datetime,
        slot_duration_minutes: int = 30,
    ) -> List[Dict[str, datetime]]:
        """
        Check calendar availability.

        Args:
            integration_id: Calendar integration ID
            organization_id: Organization ID
            calendar_ids: Calendar IDs to check
            start_time: Start of availability window
            end_time: End of availability window
            slot_duration_minutes: Minimum slot duration

        Returns:
            List of available time slots
        """
        calendar = self._registry.get_calendar_provider(integration_id)
        if not calendar:
            raise IntegrationError(f"Integration {integration_id} is not a calendar")

        available = await calendar.get_availability(
            calendar_ids=calendar_ids,
            start_time=start_time,
            end_time=end_time,
            slot_duration_minutes=slot_duration_minutes,
        )

        # Track usage
        self._track_usage(organization_id, "calendar_availability_checks", 1)

        return available

    async def schedule_meeting(
        self,
        integration_id: str,
        organization_id: str,
        title: str,
        start_time: datetime,
        end_time: datetime,
        attendee_emails: List[str],
        description: str = "",
        add_video_link: bool = True,
    ) -> CalendarEvent:
        """
        Schedule a meeting.

        Args:
            integration_id: Calendar integration ID
            organization_id: Organization ID
            title: Meeting title
            start_time: Start time
            end_time: End time
            attendee_emails: Attendee email addresses
            description: Meeting description
            add_video_link: Add video conferencing link

        Returns:
            Created calendar event
        """
        calendar = self._registry.get_calendar_provider(integration_id)
        if not calendar:
            raise IntegrationError(f"Integration {integration_id} is not a calendar")

        # Determine video link parameter based on provider
        kwargs = {
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "attendee_emails": attendee_emails,
            "description": description,
        }

        if hasattr(calendar, "schedule_meeting"):
            # Try provider-specific method
            if "google" in integration_id.lower():
                kwargs["add_meet_link"] = add_video_link
            elif "outlook" in integration_id.lower():
                kwargs["add_teams_link"] = add_video_link

            event = await calendar.schedule_meeting(**kwargs)
        else:
            # Fall back to generic event creation
            event = CalendarEvent(
                id="",
                calendar_id=None,
                title=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                attendees=[{"email": email} for email in attendee_emails],
            )
            event = await calendar.create_event(event)

        # Track usage
        self._track_usage(organization_id, "calendar_meetings_scheduled", 1)

        logger.info(
            f"Scheduled meeting via {integration_id}: {title} "
            f"({start_time} - {end_time})"
        )

        return event

    async def list_upcoming_events(
        self,
        integration_id: str,
        organization_id: str,
        calendar_id: Optional[str] = None,
        days: int = 7,
    ) -> List[CalendarEvent]:
        """
        List upcoming calendar events.

        Args:
            integration_id: Calendar integration ID
            organization_id: Organization ID
            calendar_id: Specific calendar to query
            days: Number of days to look ahead

        Returns:
            List of upcoming events
        """
        calendar = self._registry.get_calendar_provider(integration_id)
        if not calendar:
            raise IntegrationError(f"Integration {integration_id} is not a calendar")

        start_time = datetime.utcnow()
        end_time = start_time + timedelta(days=days)

        events = await calendar.list_events(
            calendar_id=calendar_id,
            start_time=start_time,
            end_time=end_time,
        )

        # Track usage
        self._track_usage(organization_id, "calendar_events_listed", 1)

        return events

    # =========================================================================
    # Storage Operations
    # =========================================================================

    async def upload_file(
        self,
        integration_id: str,
        organization_id: str,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """
        Upload a file to storage.

        Args:
            integration_id: Storage integration ID
            organization_id: Organization ID
            key: Object key/path
            data: File content
            content_type: MIME type
            metadata: Custom metadata

        Returns:
            Storage object
        """
        storage = self._registry.get_storage_provider(integration_id)
        if not storage:
            raise IntegrationError(f"Integration {integration_id} is not storage")

        obj = await storage.upload_object(
            key=key,
            data=data,
            content_type=content_type,
            metadata=metadata,
        )

        # Track usage
        self._track_usage(organization_id, "storage_uploads", 1)
        self._track_usage(organization_id, "storage_bytes_uploaded", len(data))

        logger.info(f"Uploaded file to {integration_id}: {key} ({len(data)} bytes)")

        return obj

    async def download_file(
        self,
        integration_id: str,
        organization_id: str,
        key: str,
    ) -> bytes:
        """
        Download a file from storage.

        Args:
            integration_id: Storage integration ID
            organization_id: Organization ID
            key: Object key/path

        Returns:
            File content
        """
        storage = self._registry.get_storage_provider(integration_id)
        if not storage:
            raise IntegrationError(f"Integration {integration_id} is not storage")

        data = await storage.download_object(key)

        # Track usage
        self._track_usage(organization_id, "storage_downloads", 1)
        self._track_usage(organization_id, "storage_bytes_downloaded", len(data))

        return data

    async def generate_download_url(
        self,
        integration_id: str,
        organization_id: str,
        key: str,
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a presigned download URL.

        Args:
            integration_id: Storage integration ID
            organization_id: Organization ID
            key: Object key/path
            expires_in: URL validity in seconds

        Returns:
            Presigned URL
        """
        storage = self._registry.get_storage_provider(integration_id)
        if not storage:
            raise IntegrationError(f"Integration {integration_id} is not storage")

        url = storage.generate_presigned_url(
            key=key,
            expires_in=expires_in,
            method="GET",
        )

        # Track usage
        self._track_usage(organization_id, "storage_presigned_urls", 1)

        return url

    # =========================================================================
    # Sync Management
    # =========================================================================

    def get_sync_job(self, job_id: str) -> Optional[SyncJob]:
        """Get a sync job by ID."""
        return self._sync_jobs.get(job_id)

    def list_sync_jobs(
        self,
        organization_id: str,
        integration_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[SyncJob]:
        """List sync jobs."""
        jobs = list(self._sync_jobs.values())

        jobs = [j for j in jobs if j.organization_id == organization_id]

        if integration_id:
            jobs = [j for j in jobs if j.integration_id == integration_id]
        if status:
            jobs = [j for j in jobs if j.status == status]

        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    async def cancel_sync_job(self, job_id: str) -> bool:
        """Cancel a running sync job."""
        job = self._sync_jobs.get(job_id)
        if not job or job.status != "running":
            return False

        job.status = "cancelled"
        job.completed_at = datetime.utcnow()
        return True

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_sync_complete(self, handler: Callable) -> None:
        """Register handler for sync completion events."""
        self._on_sync_complete.append(handler)

    def on_contact_synced(self, handler: Callable) -> None:
        """Register handler for contact sync events."""
        self._on_contact_synced.append(handler)

    def on_call_logged(self, handler: Callable) -> None:
        """Register handler for call logging events."""
        self._on_call_logged.append(handler)

    async def _handle_integration_connected(self, instance: IntegrationInstance) -> None:
        """Handle integration connected event."""
        logger.info(f"Integration connected: {instance.id}")

    async def _handle_integration_disconnected(self, instance: IntegrationInstance) -> None:
        """Handle integration disconnected event."""
        logger.info(f"Integration disconnected: {instance.id}")

    async def _handle_integration_error(
        self,
        instance: IntegrationInstance,
        error: Exception,
    ) -> None:
        """Handle integration error event."""
        logger.error(f"Integration error for {instance.id}: {error}")

    # =========================================================================
    # Usage Tracking
    # =========================================================================

    def _track_usage(self, organization_id: str, metric: str, value: int) -> None:
        """Track usage metric."""
        if organization_id not in self._usage_stats:
            self._usage_stats[organization_id] = {}

        if metric not in self._usage_stats[organization_id]:
            self._usage_stats[organization_id][metric] = 0

        self._usage_stats[organization_id][metric] += value

    def get_usage_stats(self, organization_id: str) -> Dict[str, int]:
        """Get usage statistics for an organization."""
        return self._usage_stats.get(organization_id, {}).copy()

    def reset_usage_stats(self, organization_id: str) -> None:
        """Reset usage statistics for an organization."""
        if organization_id in self._usage_stats:
            self._usage_stats[organization_id] = {}

    # =========================================================================
    # Provider Discovery
    # =========================================================================

    def list_available_providers(
        self,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List available integration providers.

        Args:
            category: Filter by category

        Returns:
            List of provider information
        """
        provider_category = None
        if category:
            try:
                provider_category = ProviderCategory(category)
            except ValueError:
                pass

        providers = self._registry.list_providers(category=provider_category)

        return [
            {
                "name": p.name,
                "display_name": p.display_name,
                "category": p.category.value,
                "auth_type": p.auth_type.value,
                "description": p.description,
                "capabilities": p.capabilities,
                "beta": p.beta,
            }
            for p in providers
        ]

    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed provider information."""
        provider = self._registry.get_provider(provider_name)
        if not provider:
            return None

        return {
            "name": provider.name,
            "display_name": provider.display_name,
            "category": provider.category.value,
            "auth_type": provider.auth_type.value,
            "description": provider.description,
            "icon_url": provider.icon_url,
            "documentation_url": provider.documentation_url,
            "required_credentials": provider.required_credentials,
            "optional_settings": provider.optional_settings,
            "capabilities": provider.capabilities,
            "beta": provider.beta,
        }

    # =========================================================================
    # Health and Statistics
    # =========================================================================

    async def get_integration_health(
        self,
        integration_id: str,
        organization_id: str,
    ) -> Dict[str, Any]:
        """Get health status for an integration."""
        instance = await self._registry.get_integration(integration_id, organization_id)
        if not instance:
            return {"status": "not_found"}

        return await self._registry.health_check(integration_id)

    async def get_all_integrations_health(
        self,
        organization_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Get health status for all integrations."""
        return await self._registry.health_check_all(organization_id)

    def get_service_stats(self, organization_id: Optional[str] = None) -> Dict[str, Any]:
        """Get service statistics."""
        registry_stats = self._registry.get_stats(organization_id)

        return {
            **registry_stats,
            "oauth_sessions": len(self._oauth_sessions),
            "sync_jobs": len(self._sync_jobs),
            "active_sync_jobs": len([j for j in self._sync_jobs.values() if j.status == "running"]),
        }

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> None:
        """Start the integration service."""
        # Start health monitoring
        await self._registry.start_health_monitoring(interval_seconds=300)

        logger.info("Integration service started")

    async def stop(self) -> None:
        """Stop the integration service."""
        # Stop health monitoring
        await self._registry.stop_health_monitoring()

        # Cancel any running sync jobs
        for job in self._sync_jobs.values():
            if job.status == "running":
                job.status = "cancelled"
                job.completed_at = datetime.utcnow()

        logger.info("Integration service stopped")


def create_integration_service(
    config: Optional[IntegrationServiceConfig] = None,
    credential_store: Optional[CredentialStore] = None,
) -> IntegrationService:
    """
    Create an integration service with default configuration.

    Args:
        config: Service configuration
        credential_store: Credential storage backend

    Returns:
        Configured IntegrationService
    """
    return IntegrationService(
        config=config,
        credential_store=credential_store,
    )

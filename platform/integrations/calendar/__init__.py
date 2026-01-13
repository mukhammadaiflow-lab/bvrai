"""
Calendar Integrations Package

This package provides calendar provider implementations for the voice agent platform,
enabling calendar synchronization, availability checking, and meeting scheduling
across popular calendar platforms.

Supported Providers:
- Google Calendar: Full calendar API with Google Meet integration
- Outlook/Office 365: Microsoft Graph API with Teams integration

Example usage:

    from platform.integrations.calendar import (
        GoogleCalendarProvider,
        OutlookCalendarProvider,
        get_calendar_provider,
    )

    # Create Google Calendar provider
    gcal = GoogleCalendarProvider(
        integration_id="int_123",
        organization_id="org_456",
        credentials={
            "access_token": "...",
            "refresh_token": "...",
            "client_id": "...",
            "client_secret": "...",
        },
    )

    # List events
    events = await gcal.list_events(
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow() + timedelta(days=7),
    )

    # Check availability
    available_slots = await gcal.get_availability(
        calendar_ids=["primary"],
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow() + timedelta(days=1),
        slot_duration_minutes=30,
    )

    # Schedule meeting
    meeting = await gcal.schedule_meeting(
        title="Strategy Review",
        start_time=datetime.utcnow() + timedelta(hours=2),
        end_time=datetime.utcnow() + timedelta(hours=3),
        attendee_emails=["colleague@company.com"],
        add_meet_link=True,
    )
"""

from typing import Dict, Any, Optional, Type

from ..base import CalendarProvider, IntegrationError
from .google import GoogleCalendarProvider
from .outlook import OutlookCalendarProvider


# Registry of available calendar providers
CALENDAR_PROVIDERS: Dict[str, Type[CalendarProvider]] = {
    "google_calendar": GoogleCalendarProvider,
    "google": GoogleCalendarProvider,  # Alias
    "outlook_calendar": OutlookCalendarProvider,
    "outlook": OutlookCalendarProvider,  # Alias
    "office365": OutlookCalendarProvider,  # Alias
    "microsoft": OutlookCalendarProvider,  # Alias
}


def get_calendar_provider(
    provider_name: str,
    integration_id: str,
    organization_id: str,
    credentials: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
) -> CalendarProvider:
    """
    Factory function to create a calendar provider instance.

    Args:
        provider_name: Name of the provider (google_calendar, outlook_calendar)
        integration_id: Unique integration identifier
        organization_id: Organization identifier
        credentials: Provider-specific credentials
        settings: Optional provider settings

    Returns:
        Configured calendar provider instance

    Raises:
        IntegrationError: If provider is not supported
    """
    provider_class = CALENDAR_PROVIDERS.get(provider_name.lower())
    if not provider_class:
        raise IntegrationError(
            f"Unsupported calendar provider: {provider_name}. "
            f"Available providers: google_calendar, outlook_calendar"
        )

    return provider_class(
        integration_id=integration_id,
        organization_id=organization_id,
        credentials=credentials,
        settings=settings or {},
    )


def list_calendar_providers() -> Dict[str, Dict[str, Any]]:
    """
    List all available calendar providers with their metadata.

    Returns:
        Dictionary of provider information
    """
    # Deduplicate aliases
    unique_providers = {
        "google_calendar": GoogleCalendarProvider,
        "outlook_calendar": OutlookCalendarProvider,
    }

    providers = {}
    for name, provider_class in unique_providers.items():
        providers[name] = {
            "name": provider_class.PROVIDER_NAME,
            "auth_type": provider_class.AUTH_TYPE.value,
            "capabilities": [
                "list_calendars",
                "list_events",
                "create_event",
                "update_event",
                "delete_event",
                "get_availability",
                "get_free_busy",
                "schedule_meeting",
                "find_meeting_time",
                "sync_events",
            ],
            "meeting_provider": "google_meet" if "google" in name else "teams",
        }
    return providers


__all__ = [
    # Providers
    "GoogleCalendarProvider",
    "OutlookCalendarProvider",
    # Factory
    "get_calendar_provider",
    "list_calendar_providers",
    # Registry
    "CALENDAR_PROVIDERS",
]

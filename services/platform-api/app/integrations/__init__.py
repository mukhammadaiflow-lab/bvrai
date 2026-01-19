"""External integrations module."""

from app.integrations.base import (
    IntegrationConfig,
    IntegrationStatus,
    Integration,
    IntegrationRegistry,
    get_integration_registry,
)

from app.integrations.crm import (
    CRMContact,
    CRMDeal,
    CRMIntegration,
    SalesforceIntegration,
    HubSpotIntegration,
)

from app.integrations.calendar import (
    CalendarEvent,
    CalendarSlot,
    CalendarIntegration,
    GoogleCalendarIntegration,
    OutlookCalendarIntegration,
)

from app.integrations.messaging import (
    Message,
    MessagingIntegration,
    SlackIntegration,
    TeamsIntegration,
)

__all__ = [
    # Base
    "IntegrationConfig",
    "IntegrationStatus",
    "Integration",
    "IntegrationRegistry",
    "get_integration_registry",
    # CRM
    "CRMContact",
    "CRMDeal",
    "CRMIntegration",
    "SalesforceIntegration",
    "HubSpotIntegration",
    # Calendar
    "CalendarEvent",
    "CalendarSlot",
    "CalendarIntegration",
    "GoogleCalendarIntegration",
    "OutlookCalendarIntegration",
    # Messaging
    "Message",
    "MessagingIntegration",
    "SlackIntegration",
    "TeamsIntegration",
]

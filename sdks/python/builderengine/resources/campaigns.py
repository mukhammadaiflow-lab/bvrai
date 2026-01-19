"""
Builder Engine Python SDK - Campaigns Resource

This module provides methods for managing call campaigns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List, BinaryIO

from builderengine.resources.base import BaseResource, PaginatedResponse
from builderengine.models import Campaign, CampaignStatus, CampaignContact
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class CampaignsResource(BaseResource):
    """
    Resource for managing call campaigns.

    Campaigns allow you to make batch outbound calls to a list of
    contacts. You can schedule campaigns, set calling windows, and
    track progress.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> campaign = client.campaigns.create(
        ...     name="Product Launch Outreach",
        ...     agent_id="agent_abc123",
        ...     contacts=[
        ...         {"phone_number": "+1234567890", "name": "John Doe"},
        ...         {"phone_number": "+0987654321", "name": "Jane Smith"}
        ...     ]
        ... )
        >>> client.campaigns.start(campaign.id)
    """

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[CampaignStatus] = None,
        agent_id: Optional[str] = None,
        search: Optional[str] = None,
    ) -> PaginatedResponse[Campaign]:
        """
        List all campaigns.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            status: Filter by campaign status
            agent_id: Filter by agent
            search: Search by name

        Returns:
            PaginatedResponse containing Campaign objects
        """
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            status=status.value if status else None,
            agent_id=agent_id,
            search=search,
        )
        response = self._get(Endpoints.CAMPAIGNS, params=params)
        return self._parse_paginated_response(response, Campaign)

    def get(self, campaign_id: str) -> Campaign:
        """
        Get a campaign by ID.

        Args:
            campaign_id: The campaign's unique identifier

        Returns:
            Campaign object
        """
        path = Endpoints.CAMPAIGN.format(campaign_id=campaign_id)
        response = self._get(path)
        return Campaign.from_dict(response)

    def create(
        self,
        name: str,
        agent_id: str,
        contacts: Optional[List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        max_concurrent_calls: int = 10,
        calls_per_minute: int = 5,
        max_attempts: int = 3,
        retry_delay_minutes: int = 60,
        scheduled_start: Optional[str] = None,
        scheduled_end: Optional[str] = None,
        timezone: str = "UTC",
        calling_hours_start: str = "09:00",
        calling_hours_end: str = "17:00",
        calling_days: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Campaign:
        """
        Create a new campaign.

        Args:
            name: Name of the campaign
            agent_id: Agent to use for calls
            contacts: List of contacts to call
            description: Description of the campaign
            max_concurrent_calls: Maximum simultaneous calls
            calls_per_minute: Rate of new calls per minute
            max_attempts: Maximum call attempts per contact
            retry_delay_minutes: Delay between retry attempts
            scheduled_start: Start time (ISO format)
            scheduled_end: End time (ISO format)
            timezone: Timezone for scheduling
            calling_hours_start: Daily start time (HH:MM)
            calling_hours_end: Daily end time (HH:MM)
            calling_days: Days to make calls (mon, tue, etc.)
            metadata: Custom metadata

        Returns:
            Created Campaign object

        Example:
            >>> campaign = client.campaigns.create(
            ...     name="Customer Survey",
            ...     agent_id="agent_abc123",
            ...     contacts=[
            ...         {"phone_number": "+1234567890", "name": "John", "custom_data": {"order_id": "123"}},
            ...     ],
            ...     max_concurrent_calls=5,
            ...     calling_hours_start="10:00",
            ...     calling_hours_end="18:00",
            ...     calling_days=["mon", "tue", "wed", "thu", "fri"]
            ... )
        """
        data: Dict[str, Any] = {
            "name": name,
            "agent_id": agent_id,
            "max_concurrent_calls": max_concurrent_calls,
            "calls_per_minute": calls_per_minute,
            "max_attempts": max_attempts,
            "retry_delay_minutes": retry_delay_minutes,
            "timezone": timezone,
            "calling_hours_start": calling_hours_start,
            "calling_hours_end": calling_hours_end,
        }

        if contacts:
            data["contacts"] = contacts
        if description:
            data["description"] = description
        if scheduled_start:
            data["scheduled_start"] = scheduled_start
        if scheduled_end:
            data["scheduled_end"] = scheduled_end
        if calling_days:
            data["calling_days"] = calling_days
        if metadata:
            data["metadata"] = metadata

        response = self._post(Endpoints.CAMPAIGNS, json=data)
        return Campaign.from_dict(response)

    def update(
        self,
        campaign_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_concurrent_calls: Optional[int] = None,
        calls_per_minute: Optional[int] = None,
        max_attempts: Optional[int] = None,
        scheduled_start: Optional[str] = None,
        scheduled_end: Optional[str] = None,
        calling_hours_start: Optional[str] = None,
        calling_hours_end: Optional[str] = None,
        calling_days: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Campaign:
        """
        Update a campaign.

        Note: Cannot update while campaign is running.

        Args:
            campaign_id: The campaign's unique identifier
            name: New name
            description: New description
            max_concurrent_calls: New concurrent call limit
            calls_per_minute: New call rate
            max_attempts: New max attempts
            scheduled_start: New start time
            scheduled_end: New end time
            calling_hours_start: New daily start time
            calling_hours_end: New daily end time
            calling_days: New calling days
            metadata: New metadata

        Returns:
            Updated Campaign object
        """
        data: Dict[str, Any] = {}

        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if max_concurrent_calls is not None:
            data["max_concurrent_calls"] = max_concurrent_calls
        if calls_per_minute is not None:
            data["calls_per_minute"] = calls_per_minute
        if max_attempts is not None:
            data["max_attempts"] = max_attempts
        if scheduled_start is not None:
            data["scheduled_start"] = scheduled_start
        if scheduled_end is not None:
            data["scheduled_end"] = scheduled_end
        if calling_hours_start is not None:
            data["calling_hours_start"] = calling_hours_start
        if calling_hours_end is not None:
            data["calling_hours_end"] = calling_hours_end
        if calling_days is not None:
            data["calling_days"] = calling_days
        if metadata is not None:
            data["metadata"] = metadata

        path = Endpoints.CAMPAIGN.format(campaign_id=campaign_id)
        response = self._patch(path, json=data)
        return Campaign.from_dict(response)

    def delete(self, campaign_id: str) -> None:
        """
        Delete a campaign.

        Note: Cannot delete while campaign is running.

        Args:
            campaign_id: The campaign's unique identifier
        """
        path = Endpoints.CAMPAIGN.format(campaign_id=campaign_id)
        self._delete(path)

    def start(self, campaign_id: str) -> Campaign:
        """
        Start a campaign.

        Args:
            campaign_id: The campaign's unique identifier

        Returns:
            Updated Campaign object
        """
        path = Endpoints.CAMPAIGN_START.format(campaign_id=campaign_id)
        response = self._post(path)
        return Campaign.from_dict(response)

    def pause(self, campaign_id: str) -> Campaign:
        """
        Pause a running campaign.

        Args:
            campaign_id: The campaign's unique identifier

        Returns:
            Updated Campaign object
        """
        path = Endpoints.CAMPAIGN_PAUSE.format(campaign_id=campaign_id)
        response = self._post(path)
        return Campaign.from_dict(response)

    def resume(self, campaign_id: str) -> Campaign:
        """
        Resume a paused campaign.

        Args:
            campaign_id: The campaign's unique identifier

        Returns:
            Updated Campaign object
        """
        path = Endpoints.CAMPAIGN_RESUME.format(campaign_id=campaign_id)
        response = self._post(path)
        return Campaign.from_dict(response)

    def cancel(self, campaign_id: str) -> Campaign:
        """
        Cancel a campaign.

        This stops all pending calls and marks the campaign as canceled.

        Args:
            campaign_id: The campaign's unique identifier

        Returns:
            Updated Campaign object
        """
        path = Endpoints.CAMPAIGN_CANCEL.format(campaign_id=campaign_id)
        response = self._post(path)
        return Campaign.from_dict(response)

    # Contact methods

    def list_contacts(
        self,
        campaign_id: str,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
    ) -> PaginatedResponse[CampaignContact]:
        """
        List contacts in a campaign.

        Args:
            campaign_id: The campaign's unique identifier
            page: Page number
            page_size: Items per page
            status: Filter by status (pending, called, completed, failed, skipped)

        Returns:
            PaginatedResponse containing CampaignContact objects
        """
        path = Endpoints.CAMPAIGN_CONTACTS.format(campaign_id=campaign_id)
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            status=status,
        )
        response = self._get(path, params=params)
        return self._parse_paginated_response(response, CampaignContact)

    def add_contacts(
        self,
        campaign_id: str,
        contacts: List[Dict[str, Any]],
    ) -> List[CampaignContact]:
        """
        Add contacts to a campaign.

        Args:
            campaign_id: The campaign's unique identifier
            contacts: List of contact data

        Returns:
            List of created CampaignContact objects

        Example:
            >>> contacts = client.campaigns.add_contacts(
            ...     campaign_id="campaign_abc123",
            ...     contacts=[
            ...         {"phone_number": "+1111111111", "name": "Alice"},
            ...         {"phone_number": "+2222222222", "name": "Bob"}
            ...     ]
            ... )
        """
        path = Endpoints.CAMPAIGN_CONTACTS.format(campaign_id=campaign_id)
        response = self._post(path, json={"contacts": contacts})
        return [CampaignContact.from_dict(c) for c in response.get("contacts", [])]

    def get_contact(self, campaign_id: str, contact_id: str) -> CampaignContact:
        """
        Get a contact by ID.

        Args:
            campaign_id: The campaign's unique identifier
            contact_id: The contact's unique identifier

        Returns:
            CampaignContact object
        """
        path = Endpoints.CAMPAIGN_CONTACT.format(
            campaign_id=campaign_id,
            contact_id=contact_id
        )
        response = self._get(path)
        return CampaignContact.from_dict(response)

    def update_contact(
        self,
        campaign_id: str,
        contact_id: str,
        status: Optional[str] = None,
        scheduled_at: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ) -> CampaignContact:
        """
        Update a contact.

        Args:
            campaign_id: The campaign's unique identifier
            contact_id: The contact's unique identifier
            status: New status
            scheduled_at: New scheduled time
            custom_data: New custom data

        Returns:
            Updated CampaignContact object
        """
        data: Dict[str, Any] = {}

        if status is not None:
            data["status"] = status
        if scheduled_at is not None:
            data["scheduled_at"] = scheduled_at
        if custom_data is not None:
            data["custom_data"] = custom_data

        path = Endpoints.CAMPAIGN_CONTACT.format(
            campaign_id=campaign_id,
            contact_id=contact_id
        )
        response = self._patch(path, json=data)
        return CampaignContact.from_dict(response)

    def remove_contact(self, campaign_id: str, contact_id: str) -> None:
        """
        Remove a contact from a campaign.

        Args:
            campaign_id: The campaign's unique identifier
            contact_id: The contact's unique identifier
        """
        path = Endpoints.CAMPAIGN_CONTACT.format(
            campaign_id=campaign_id,
            contact_id=contact_id
        )
        self._delete(path)

    def import_contacts(
        self,
        campaign_id: str,
        file: BinaryIO,
        mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Import contacts from a CSV file.

        Args:
            campaign_id: The campaign's unique identifier
            file: CSV file object
            mapping: Column mapping (e.g., {"Phone": "phone_number", "Name": "name"})

        Returns:
            Import result with success/error counts

        Example:
            >>> with open("contacts.csv", "rb") as f:
            ...     result = client.campaigns.import_contacts(
            ...         campaign_id="campaign_abc123",
            ...         file=f,
            ...         mapping={"Phone Number": "phone_number", "Full Name": "name"}
            ...     )
            >>> print(f"Imported {result['imported']} contacts")
        """
        path = Endpoints.CAMPAIGN_IMPORT_CONTACTS.format(campaign_id=campaign_id)
        files = {"file": file}
        data = {}
        if mapping:
            data["mapping"] = mapping
        return self._post(path, data=data, files=files)

    def get_analytics(
        self,
        campaign_id: str,
    ) -> Dict[str, Any]:
        """
        Get analytics for a campaign.

        Args:
            campaign_id: The campaign's unique identifier

        Returns:
            Analytics data including completion rates, durations, etc.
        """
        path = Endpoints.CAMPAIGN_ANALYTICS.format(campaign_id=campaign_id)
        return self._get(path)

    def skip_contact(self, campaign_id: str, contact_id: str, reason: Optional[str] = None) -> CampaignContact:
        """
        Skip a contact (won't be called).

        Args:
            campaign_id: The campaign's unique identifier
            contact_id: The contact's unique identifier
            reason: Reason for skipping

        Returns:
            Updated CampaignContact object
        """
        return self.update_contact(
            campaign_id,
            contact_id,
            status="skipped",
            custom_data={"skip_reason": reason} if reason else None
        )

    def retry_contact(self, campaign_id: str, contact_id: str) -> CampaignContact:
        """
        Reset a contact to be called again.

        Args:
            campaign_id: The campaign's unique identifier
            contact_id: The contact's unique identifier

        Returns:
            Updated CampaignContact object
        """
        return self.update_contact(
            campaign_id,
            contact_id,
            status="pending"
        )

    def duplicate(self, campaign_id: str, name: Optional[str] = None) -> Campaign:
        """
        Duplicate a campaign.

        Args:
            campaign_id: The campaign's unique identifier
            name: Name for the new campaign

        Returns:
            New Campaign object
        """
        path = f"{Endpoints.CAMPAIGN.format(campaign_id=campaign_id)}/duplicate"
        data = {}
        if name:
            data["name"] = name
        response = self._post(path, json=data)
        return Campaign.from_dict(response)

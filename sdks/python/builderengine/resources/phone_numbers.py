"""
Builder Engine Python SDK - Phone Numbers Resource

This module provides methods for managing phone numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List

from builderengine.resources.base import BaseResource, PaginatedResponse
from builderengine.models import PhoneNumber, PhoneNumberType, PhoneNumberCapability
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class PhoneNumbersResource(BaseResource):
    """
    Resource for managing phone numbers.

    Phone numbers are the entry points for inbound calls and are used
    as the caller ID for outbound calls. This resource provides methods
    for purchasing, configuring, and managing phone numbers.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> # Search for available numbers
        >>> available = client.phone_numbers.search_available(
        ...     country="US",
        ...     area_code="415"
        ... )
        >>> # Purchase a number
        >>> number = client.phone_numbers.purchase(available[0]["number"])
    """

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        type: Optional[PhoneNumberType] = None,
        agent_id: Optional[str] = None,
        search: Optional[str] = None,
    ) -> PaginatedResponse[PhoneNumber]:
        """
        List all phone numbers.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            status: Filter by status (active, inactive, pending)
            type: Filter by number type
            agent_id: Filter by assigned agent
            search: Search by number or friendly name

        Returns:
            PaginatedResponse containing PhoneNumber objects
        """
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            status=status,
            type=type.value if type else None,
            agent_id=agent_id,
            search=search,
        )
        response = self._get(Endpoints.PHONE_NUMBERS, params=params)
        return self._parse_paginated_response(response, PhoneNumber)

    def get(self, phone_number_id: str) -> PhoneNumber:
        """
        Get a phone number by ID.

        Args:
            phone_number_id: The phone number's unique identifier

        Returns:
            PhoneNumber object
        """
        path = Endpoints.PHONE_NUMBER.format(phone_number_id=phone_number_id)
        response = self._get(path)
        return PhoneNumber.from_dict(response)

    def search_available(
        self,
        country: str = "US",
        type: PhoneNumberType = PhoneNumberType.LOCAL,
        area_code: Optional[str] = None,
        contains: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        capabilities: Optional[List[PhoneNumberCapability]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search for available phone numbers to purchase.

        Args:
            country: ISO country code (e.g., "US", "GB")
            type: Type of number (local, toll_free, mobile)
            area_code: Filter by area code
            contains: Pattern to match (e.g., "555")
            city: Filter by city
            state: Filter by state/region
            capabilities: Required capabilities
            limit: Maximum results to return

        Returns:
            List of available numbers with pricing

        Example:
            >>> numbers = client.phone_numbers.search_available(
            ...     country="US",
            ...     area_code="415",
            ...     capabilities=[PhoneNumberCapability.VOICE, PhoneNumberCapability.SMS]
            ... )
            >>> for num in numbers:
            ...     print(f"{num['number']} - ${num['monthly_price']}/mo")
        """
        params = {
            "country": country,
            "type": type.value,
            "limit": limit,
        }
        if area_code:
            params["area_code"] = area_code
        if contains:
            params["contains"] = contains
        if city:
            params["city"] = city
        if state:
            params["state"] = state
        if capabilities:
            params["capabilities"] = ",".join(c.value for c in capabilities)

        response = self._get(Endpoints.PHONE_NUMBERS_AVAILABLE, params=params)
        return response.get("numbers", [])

    def purchase(
        self,
        number: str,
        friendly_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        voice_url: Optional[str] = None,
        sms_url: Optional[str] = None,
        status_callback_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PhoneNumber:
        """
        Purchase a phone number.

        Args:
            number: The phone number to purchase (E.164 format)
            friendly_name: Display name for the number
            agent_id: Agent to assign the number to
            voice_url: Webhook URL for incoming calls
            sms_url: Webhook URL for incoming SMS
            status_callback_url: URL for status updates
            metadata: Custom metadata

        Returns:
            Purchased PhoneNumber object

        Example:
            >>> number = client.phone_numbers.purchase(
            ...     number="+14155551234",
            ...     friendly_name="Sales Hotline",
            ...     agent_id="agent_abc123"
            ... )
        """
        data: Dict[str, Any] = {"number": number}

        if friendly_name:
            data["friendly_name"] = friendly_name
        if agent_id:
            data["agent_id"] = agent_id
        if voice_url:
            data["voice_url"] = voice_url
        if sms_url:
            data["sms_url"] = sms_url
        if status_callback_url:
            data["status_callback_url"] = status_callback_url
        if metadata:
            data["metadata"] = metadata

        response = self._post(Endpoints.PHONE_NUMBER_PURCHASE, json=data)
        return PhoneNumber.from_dict(response)

    def update(
        self,
        phone_number_id: str,
        friendly_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        voice_url: Optional[str] = None,
        sms_url: Optional[str] = None,
        status_callback_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PhoneNumber:
        """
        Update a phone number's configuration.

        Args:
            phone_number_id: The phone number's unique identifier
            friendly_name: New display name
            agent_id: New agent assignment (use "" to unassign)
            voice_url: New voice webhook URL
            sms_url: New SMS webhook URL
            status_callback_url: New status callback URL
            metadata: New metadata (merged with existing)

        Returns:
            Updated PhoneNumber object
        """
        data: Dict[str, Any] = {}

        if friendly_name is not None:
            data["friendly_name"] = friendly_name
        if agent_id is not None:
            data["agent_id"] = agent_id
        if voice_url is not None:
            data["voice_url"] = voice_url
        if sms_url is not None:
            data["sms_url"] = sms_url
        if status_callback_url is not None:
            data["status_callback_url"] = status_callback_url
        if metadata is not None:
            data["metadata"] = metadata

        path = Endpoints.PHONE_NUMBER.format(phone_number_id=phone_number_id)
        response = self._patch(path, json=data)
        return PhoneNumber.from_dict(response)

    def release(self, phone_number_id: str) -> None:
        """
        Release a phone number.

        This will permanently release the number and stop all charges.
        The number may become available to other users.

        Args:
            phone_number_id: The phone number's unique identifier
        """
        path = Endpoints.PHONE_NUMBER_RELEASE.format(phone_number_id=phone_number_id)
        self._post(path)

    def delete(self, phone_number_id: str) -> None:
        """
        Delete (release) a phone number.

        Alias for release() method.

        Args:
            phone_number_id: The phone number's unique identifier
        """
        self.release(phone_number_id)

    def assign_agent(self, phone_number_id: str, agent_id: str) -> PhoneNumber:
        """
        Assign an agent to a phone number.

        Args:
            phone_number_id: The phone number's unique identifier
            agent_id: The agent's unique identifier

        Returns:
            Updated PhoneNumber object

        Example:
            >>> number = client.phone_numbers.assign_agent(
            ...     phone_number_id="pn_abc123",
            ...     agent_id="agent_xyz789"
            ... )
        """
        return self.update(phone_number_id, agent_id=agent_id)

    def unassign_agent(self, phone_number_id: str) -> PhoneNumber:
        """
        Unassign the agent from a phone number.

        Args:
            phone_number_id: The phone number's unique identifier

        Returns:
            Updated PhoneNumber object
        """
        return self.update(phone_number_id, agent_id="")

    def get_usage(
        self,
        phone_number_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get usage statistics for a phone number.

        Args:
            phone_number_id: The phone number's unique identifier
            start_date: Start date for the period
            end_date: End date for the period

        Returns:
            Usage statistics including call counts and costs
        """
        path = f"{Endpoints.PHONE_NUMBER.format(phone_number_id=phone_number_id)}/usage"
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._get(path, params=params)

    def import_number(
        self,
        number: str,
        provider: str,
        provider_config: Dict[str, Any],
        friendly_name: Optional[str] = None,
    ) -> PhoneNumber:
        """
        Import an existing phone number from another provider.

        Args:
            number: The phone number (E.164 format)
            provider: The provider name (twilio, vonage, etc.)
            provider_config: Provider-specific configuration
            friendly_name: Display name for the number

        Returns:
            Imported PhoneNumber object
        """
        data = {
            "number": number,
            "provider": provider,
            "provider_config": provider_config,
        }
        if friendly_name:
            data["friendly_name"] = friendly_name

        response = self._post(f"{Endpoints.PHONE_NUMBERS}/import", json=data)
        return PhoneNumber.from_dict(response)

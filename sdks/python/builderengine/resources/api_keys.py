"""
Builder Engine Python SDK - API Keys Resource

This module provides methods for managing API keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List

from builderengine.resources.base import BaseResource, PaginatedResponse
from builderengine.models import APIKey
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class APIKeysResource(BaseResource):
    """
    Resource for managing API keys.

    API keys are used to authenticate requests to the Builder Engine API.
    You can create multiple keys with different permissions for different
    use cases.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> key = client.api_keys.create(
        ...     name="Production API Key",
        ...     permissions=["calls:read", "calls:write", "agents:read"]
        ... )
        >>> print(f"New key: {key['key']}")  # Only shown once!
    """

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedResponse[APIKey]:
        """
        List all API keys.

        Note: The full key is not returned, only the prefix.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            PaginatedResponse containing APIKey objects
        """
        params = self._build_pagination_params(page=page, page_size=page_size)
        response = self._get(Endpoints.API_KEYS, params=params)
        return self._parse_paginated_response(response, APIKey)

    def get(self, api_key_id: str) -> APIKey:
        """
        Get an API key by ID.

        Note: The full key is not returned, only the prefix.

        Args:
            api_key_id: The API key's unique identifier

        Returns:
            APIKey object
        """
        path = Endpoints.API_KEY.format(api_key_id=api_key_id)
        response = self._get(path)
        return APIKey.from_dict(response)

    def create(
        self,
        name: str,
        permissions: Optional[List[str]] = None,
        rate_limit: Optional[int] = None,
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new API key.

        Important: The full key is only returned once during creation.
        Make sure to save it securely.

        Args:
            name: Name/description for the key
            permissions: List of permission scopes
            rate_limit: Custom rate limit (requests/minute)
            expires_at: Expiration date (ISO format)

        Returns:
            Created key data including the full key

        Example:
            >>> result = client.api_keys.create(
            ...     name="Mobile App Key",
            ...     permissions=["calls:read", "calls:write"],
            ...     rate_limit=100,
            ...     expires_at="2025-12-31T23:59:59Z"
            ... )
            >>> print(f"Key: {result['key']}")  # Save this!
            >>> print(f"ID: {result['id']}")

        Permission Scopes:
            - agents:read - Read agent data
            - agents:write - Create/update agents
            - agents:delete - Delete agents
            - calls:read - Read call data
            - calls:write - Create/manage calls
            - phone_numbers:read - Read phone numbers
            - phone_numbers:write - Purchase/manage numbers
            - analytics:read - Access analytics
            - webhooks:read - Read webhooks
            - webhooks:write - Create/manage webhooks
            - billing:read - View billing info
            - billing:write - Manage billing
            - organization:read - Read org settings
            - organization:write - Manage org settings
            - * - Full access (admin only)
        """
        data: Dict[str, Any] = {"name": name}

        if permissions:
            data["permissions"] = permissions
        if rate_limit:
            data["rate_limit"] = rate_limit
        if expires_at:
            data["expires_at"] = expires_at

        return self._post(Endpoints.API_KEYS, json=data)

    def update(
        self,
        api_key_id: str,
        name: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        rate_limit: Optional[int] = None,
        expires_at: Optional[str] = None,
    ) -> APIKey:
        """
        Update an API key.

        Args:
            api_key_id: The API key's unique identifier
            name: New name
            permissions: New permissions
            rate_limit: New rate limit
            expires_at: New expiration date

        Returns:
            Updated APIKey object
        """
        data: Dict[str, Any] = {}

        if name is not None:
            data["name"] = name
        if permissions is not None:
            data["permissions"] = permissions
        if rate_limit is not None:
            data["rate_limit"] = rate_limit
        if expires_at is not None:
            data["expires_at"] = expires_at

        path = Endpoints.API_KEY.format(api_key_id=api_key_id)
        response = self._patch(path, json=data)
        return APIKey.from_dict(response)

    def delete(self, api_key_id: str) -> None:
        """
        Delete an API key.

        Warning: This immediately invalidates the key.

        Args:
            api_key_id: The API key's unique identifier
        """
        path = Endpoints.API_KEY.format(api_key_id=api_key_id)
        self._delete(path)

    def regenerate(self, api_key_id: str) -> Dict[str, Any]:
        """
        Regenerate an API key.

        This creates a new key value while keeping the same ID
        and settings. The old key is immediately invalidated.

        Args:
            api_key_id: The API key's unique identifier

        Returns:
            New key data including the full key

        Example:
            >>> result = client.api_keys.regenerate("key_abc123")
            >>> print(f"New key: {result['key']}")  # Save this!
        """
        path = Endpoints.API_KEY_REGENERATE.format(api_key_id=api_key_id)
        return self._post(path)

    def get_usage(
        self,
        api_key_id: str,
        period: str = "month",
    ) -> Dict[str, Any]:
        """
        Get usage statistics for an API key.

        Args:
            api_key_id: The API key's unique identifier
            period: Time period (day, week, month)

        Returns:
            Usage statistics
        """
        path = f"{Endpoints.API_KEY.format(api_key_id=api_key_id)}/usage"
        return self._get(path, params={"period": period})

    def verify(self, key: str) -> Dict[str, Any]:
        """
        Verify an API key.

        Args:
            key: The full API key to verify

        Returns:
            Key information if valid

        Example:
            >>> result = client.api_keys.verify("be_live_abc123...")
            >>> if result["valid"]:
            ...     print(f"Key belongs to: {result['organization_name']}")
        """
        return self._post(f"{Endpoints.API_KEYS}/verify", json={"key": key})

    @staticmethod
    def get_permission_scopes() -> List[Dict[str, Any]]:
        """
        Get all available permission scopes.

        Returns:
            List of permission scopes with descriptions
        """
        return [
            {"scope": "agents:read", "description": "Read agent data"},
            {"scope": "agents:write", "description": "Create and update agents"},
            {"scope": "agents:delete", "description": "Delete agents"},
            {"scope": "calls:read", "description": "Read call data and recordings"},
            {"scope": "calls:write", "description": "Create and manage calls"},
            {"scope": "phone_numbers:read", "description": "Read phone numbers"},
            {"scope": "phone_numbers:write", "description": "Purchase and manage phone numbers"},
            {"scope": "voices:read", "description": "Read voice configurations"},
            {"scope": "voices:write", "description": "Create and manage voices"},
            {"scope": "webhooks:read", "description": "Read webhook configurations"},
            {"scope": "webhooks:write", "description": "Create and manage webhooks"},
            {"scope": "knowledge_base:read", "description": "Read knowledge bases"},
            {"scope": "knowledge_base:write", "description": "Create and manage knowledge bases"},
            {"scope": "workflows:read", "description": "Read workflows"},
            {"scope": "workflows:write", "description": "Create and manage workflows"},
            {"scope": "campaigns:read", "description": "Read campaigns"},
            {"scope": "campaigns:write", "description": "Create and manage campaigns"},
            {"scope": "analytics:read", "description": "Access analytics data"},
            {"scope": "billing:read", "description": "View billing information"},
            {"scope": "billing:write", "description": "Manage billing and subscriptions"},
            {"scope": "organization:read", "description": "Read organization settings"},
            {"scope": "organization:write", "description": "Manage organization settings"},
            {"scope": "users:read", "description": "Read user data"},
            {"scope": "users:write", "description": "Manage users"},
            {"scope": "*", "description": "Full access (admin only)"},
        ]

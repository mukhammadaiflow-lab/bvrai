"""
Builder Engine Python SDK - Organizations Resource

This module provides methods for managing organizations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List

from builderengine.resources.base import BaseResource, PaginatedResponse
from builderengine.models import Organization, User
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class OrganizationsResource(BaseResource):
    """
    Resource for managing organizations.

    Organizations are the top-level container for all resources in
    Builder Engine. Each organization has its own users, agents,
    phone numbers, and billing.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> org = client.organizations.get_current()
        >>> print(f"Organization: {org.name}")
    """

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedResponse[Organization]:
        """
        List all organizations the user has access to.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            PaginatedResponse containing Organization objects
        """
        params = self._build_pagination_params(page=page, page_size=page_size)
        response = self._get(Endpoints.ORGANIZATIONS, params=params)
        return self._parse_paginated_response(response, Organization)

    def get(self, organization_id: str) -> Organization:
        """
        Get an organization by ID.

        Args:
            organization_id: The organization's unique identifier

        Returns:
            Organization object
        """
        path = Endpoints.ORGANIZATION.format(organization_id=organization_id)
        response = self._get(path)
        return Organization.from_dict(response)

    def get_current(self) -> Organization:
        """
        Get the current organization.

        Returns:
            Organization object for the current context
        """
        response = self._get(f"{Endpoints.ORGANIZATIONS}/current")
        return Organization.from_dict(response)

    def create(
        self,
        name: str,
        slug: Optional[str] = None,
        billing_email: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Organization:
        """
        Create a new organization.

        Args:
            name: Name of the organization
            slug: URL-friendly identifier (auto-generated if not provided)
            billing_email: Email for billing notifications
            settings: Organization settings
            metadata: Custom metadata

        Returns:
            Created Organization object

        Example:
            >>> org = client.organizations.create(
            ...     name="Acme Corp",
            ...     billing_email="billing@acme.com"
            ... )
        """
        data: Dict[str, Any] = {"name": name}

        if slug:
            data["slug"] = slug
        if billing_email:
            data["billing_email"] = billing_email
        if settings:
            data["settings"] = settings
        if metadata:
            data["metadata"] = metadata

        response = self._post(Endpoints.ORGANIZATIONS, json=data)
        return Organization.from_dict(response)

    def update(
        self,
        organization_id: str,
        name: Optional[str] = None,
        billing_email: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Organization:
        """
        Update an organization.

        Args:
            organization_id: The organization's unique identifier
            name: New name
            billing_email: New billing email
            settings: New settings (merged with existing)
            metadata: New metadata (merged with existing)

        Returns:
            Updated Organization object
        """
        data: Dict[str, Any] = {}

        if name is not None:
            data["name"] = name
        if billing_email is not None:
            data["billing_email"] = billing_email
        if settings is not None:
            data["settings"] = settings
        if metadata is not None:
            data["metadata"] = metadata

        path = Endpoints.ORGANIZATION.format(organization_id=organization_id)
        response = self._patch(path, json=data)
        return Organization.from_dict(response)

    def delete(self, organization_id: str) -> None:
        """
        Delete an organization.

        Warning: This permanently deletes the organization and all
        associated data. This action cannot be undone.

        Args:
            organization_id: The organization's unique identifier
        """
        path = Endpoints.ORGANIZATION.format(organization_id=organization_id)
        self._delete(path)

    # Member management

    def list_members(
        self,
        organization_id: str,
        page: int = 1,
        page_size: int = 20,
        role: Optional[str] = None,
    ) -> PaginatedResponse[User]:
        """
        List organization members.

        Args:
            organization_id: The organization's unique identifier
            page: Page number
            page_size: Items per page
            role: Filter by role (owner, admin, member, viewer)

        Returns:
            PaginatedResponse containing User objects
        """
        path = Endpoints.ORGANIZATION_MEMBERS.format(organization_id=organization_id)
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            role=role,
        )
        response = self._get(path, params=params)
        return self._parse_paginated_response(response, User)

    def get_member(self, organization_id: str, user_id: str) -> User:
        """
        Get a member by ID.

        Args:
            organization_id: The organization's unique identifier
            user_id: The user's unique identifier

        Returns:
            User object
        """
        path = Endpoints.ORGANIZATION_MEMBER.format(
            organization_id=organization_id,
            user_id=user_id
        )
        response = self._get(path)
        return User.from_dict(response)

    def update_member_role(
        self,
        organization_id: str,
        user_id: str,
        role: str,
    ) -> User:
        """
        Update a member's role.

        Args:
            organization_id: The organization's unique identifier
            user_id: The user's unique identifier
            role: New role (admin, member, viewer)

        Returns:
            Updated User object
        """
        path = Endpoints.ORGANIZATION_MEMBER.format(
            organization_id=organization_id,
            user_id=user_id
        )
        response = self._patch(path, json={"role": role})
        return User.from_dict(response)

    def remove_member(self, organization_id: str, user_id: str) -> None:
        """
        Remove a member from the organization.

        Args:
            organization_id: The organization's unique identifier
            user_id: The user's unique identifier
        """
        path = Endpoints.ORGANIZATION_MEMBER.format(
            organization_id=organization_id,
            user_id=user_id
        )
        self._delete(path)

    def invite_member(
        self,
        organization_id: str,
        email: str,
        role: str = "member",
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Invite a new member to the organization.

        Args:
            organization_id: The organization's unique identifier
            email: Email address to invite
            role: Role for the new member
            message: Custom invitation message

        Returns:
            Invitation details

        Example:
            >>> invite = client.organizations.invite_member(
            ...     organization_id="org_abc123",
            ...     email="john@example.com",
            ...     role="member"
            ... )
            >>> print(f"Invitation sent to {invite['email']}")
        """
        path = Endpoints.ORGANIZATION_INVITE.format(organization_id=organization_id)
        data = {
            "email": email,
            "role": role,
        }
        if message:
            data["message"] = message

        return self._post(path, json=data)

    def get_pending_invites(
        self,
        organization_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get pending invitations.

        Args:
            organization_id: The organization's unique identifier

        Returns:
            List of pending invitations
        """
        path = f"{Endpoints.ORGANIZATION_INVITE.format(organization_id=organization_id)}/pending"
        response = self._get(path)
        return response.get("invites", [])

    def cancel_invite(self, organization_id: str, invite_id: str) -> None:
        """
        Cancel a pending invitation.

        Args:
            organization_id: The organization's unique identifier
            invite_id: The invitation's unique identifier
        """
        path = f"{Endpoints.ORGANIZATION_INVITE.format(organization_id=organization_id)}/{invite_id}"
        self._delete(path)

    def resend_invite(self, organization_id: str, invite_id: str) -> Dict[str, Any]:
        """
        Resend an invitation email.

        Args:
            organization_id: The organization's unique identifier
            invite_id: The invitation's unique identifier

        Returns:
            Updated invitation details
        """
        path = f"{Endpoints.ORGANIZATION_INVITE.format(organization_id=organization_id)}/{invite_id}/resend"
        return self._post(path)

    # Settings

    def get_settings(self, organization_id: str) -> Dict[str, Any]:
        """
        Get organization settings.

        Args:
            organization_id: The organization's unique identifier

        Returns:
            Organization settings
        """
        path = Endpoints.ORGANIZATION_SETTINGS.format(organization_id=organization_id)
        return self._get(path)

    def update_settings(
        self,
        organization_id: str,
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update organization settings.

        Args:
            organization_id: The organization's unique identifier
            settings: Settings to update (merged with existing)

        Returns:
            Updated settings
        """
        path = Endpoints.ORGANIZATION_SETTINGS.format(organization_id=organization_id)
        return self._patch(path, json=settings)

    def transfer_ownership(
        self,
        organization_id: str,
        new_owner_id: str,
    ) -> Organization:
        """
        Transfer organization ownership to another member.

        Args:
            organization_id: The organization's unique identifier
            new_owner_id: User ID of the new owner

        Returns:
            Updated Organization object
        """
        path = f"{Endpoints.ORGANIZATION.format(organization_id=organization_id)}/transfer-ownership"
        response = self._post(path, json={"new_owner_id": new_owner_id})
        return Organization.from_dict(response)

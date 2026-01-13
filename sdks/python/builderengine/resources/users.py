"""
Builder Engine Python SDK - Users Resource

This module provides methods for managing users.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List

from builderengine.resources.base import BaseResource
from builderengine.models import User
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class UsersResource(BaseResource):
    """
    Resource for managing user accounts.

    This resource provides methods for managing the current user's
    profile, preferences, and account settings.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> user = client.users.get_me()
        >>> print(f"Logged in as: {user.email}")
    """

    def get_me(self) -> User:
        """
        Get the current user.

        Returns:
            User object for the authenticated user
        """
        response = self._get(Endpoints.USER_ME)
        return User.from_dict(response)

    def get(self, user_id: str) -> User:
        """
        Get a user by ID.

        Args:
            user_id: The user's unique identifier

        Returns:
            User object
        """
        path = Endpoints.USER.format(user_id=user_id)
        response = self._get(path)
        return User.from_dict(response)

    def update_profile(
        self,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        timezone: Optional[str] = None,
        avatar_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> User:
        """
        Update the current user's profile.

        Args:
            name: Display name
            phone: Phone number
            timezone: Timezone (e.g., "America/New_York")
            avatar_url: Avatar image URL
            metadata: Custom metadata

        Returns:
            Updated User object

        Example:
            >>> user = client.users.update_profile(
            ...     name="John Doe",
            ...     timezone="America/New_York"
            ... )
        """
        data: Dict[str, Any] = {}

        if name is not None:
            data["name"] = name
        if phone is not None:
            data["phone"] = phone
        if timezone is not None:
            data["timezone"] = timezone
        if avatar_url is not None:
            data["avatar_url"] = avatar_url
        if metadata is not None:
            data["metadata"] = metadata

        response = self._patch(Endpoints.USER_PROFILE, json=data)
        return User.from_dict(response)

    def change_password(
        self,
        current_password: str,
        new_password: str,
    ) -> Dict[str, Any]:
        """
        Change the current user's password.

        Args:
            current_password: Current password
            new_password: New password

        Returns:
            Success confirmation

        Example:
            >>> client.users.change_password(
            ...     current_password="old_password",
            ...     new_password="new_secure_password"
            ... )
        """
        return self._post(Endpoints.USER_PASSWORD, json={
            "current_password": current_password,
            "new_password": new_password,
        })

    def get_notification_preferences(self) -> Dict[str, Any]:
        """
        Get notification preferences.

        Returns:
            Notification settings
        """
        return self._get(Endpoints.USER_NOTIFICATIONS)

    def update_notification_preferences(
        self,
        email_notifications: Optional[bool] = None,
        sms_notifications: Optional[bool] = None,
        push_notifications: Optional[bool] = None,
        notification_types: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Update notification preferences.

        Args:
            email_notifications: Enable email notifications
            sms_notifications: Enable SMS notifications
            push_notifications: Enable push notifications
            notification_types: Enable/disable specific notification types

        Returns:
            Updated notification settings

        Example:
            >>> prefs = client.users.update_notification_preferences(
            ...     email_notifications=True,
            ...     notification_types={
            ...         "call_completed": True,
            ...         "campaign_completed": True,
            ...         "billing_alerts": True
            ...     }
            ... )
        """
        data: Dict[str, Any] = {}

        if email_notifications is not None:
            data["email_notifications"] = email_notifications
        if sms_notifications is not None:
            data["sms_notifications"] = sms_notifications
        if push_notifications is not None:
            data["push_notifications"] = push_notifications
        if notification_types is not None:
            data["notification_types"] = notification_types

        return self._patch(Endpoints.USER_NOTIFICATIONS, json=data)

    def get_activity_log(
        self,
        page: int = 1,
        page_size: int = 20,
        action_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get user activity log.

        Args:
            page: Page number
            page_size: Items per page
            action_type: Filter by action type
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            Paginated activity log
        """
        params = {
            "page": page,
            "page_size": page_size,
        }
        if action_type:
            params["action_type"] = action_type
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return self._get(f"{Endpoints.USER_ME}/activity", params=params)

    def get_sessions(self) -> List[Dict[str, Any]]:
        """
        Get active sessions.

        Returns:
            List of active sessions
        """
        response = self._get(f"{Endpoints.USER_ME}/sessions")
        return response.get("sessions", [])

    def revoke_session(self, session_id: str) -> None:
        """
        Revoke a session.

        Args:
            session_id: The session's unique identifier
        """
        self._delete(f"{Endpoints.USER_ME}/sessions/{session_id}")

    def revoke_all_sessions(self) -> None:
        """
        Revoke all sessions except the current one.
        """
        self._delete(f"{Endpoints.USER_ME}/sessions")

    def enable_two_factor(self) -> Dict[str, Any]:
        """
        Enable two-factor authentication.

        Returns:
            Setup data including QR code and backup codes
        """
        return self._post(f"{Endpoints.USER_ME}/2fa/enable")

    def disable_two_factor(self, code: str) -> Dict[str, Any]:
        """
        Disable two-factor authentication.

        Args:
            code: Current 2FA code

        Returns:
            Confirmation
        """
        return self._post(f"{Endpoints.USER_ME}/2fa/disable", json={"code": code})

    def verify_two_factor(self, code: str) -> Dict[str, Any]:
        """
        Verify two-factor setup.

        Args:
            code: 2FA code from authenticator app

        Returns:
            Confirmation with backup codes
        """
        return self._post(f"{Endpoints.USER_ME}/2fa/verify", json={"code": code})

    def get_backup_codes(self) -> List[str]:
        """
        Get backup codes for 2FA.

        Returns:
            List of backup codes
        """
        response = self._get(f"{Endpoints.USER_ME}/2fa/backup-codes")
        return response.get("backup_codes", [])

    def regenerate_backup_codes(self) -> List[str]:
        """
        Regenerate 2FA backup codes.

        Returns:
            New list of backup codes
        """
        response = self._post(f"{Endpoints.USER_ME}/2fa/backup-codes/regenerate")
        return response.get("backup_codes", [])

    def delete_account(self, password: str, confirmation: str = "DELETE") -> None:
        """
        Delete the user account.

        Warning: This permanently deletes the account and all
        associated data. This action cannot be undone.

        Args:
            password: Current password for confirmation
            confirmation: Must be "DELETE" to confirm
        """
        self._delete(Endpoints.USER_ME, params={
            "password": password,
            "confirmation": confirmation,
        })

    def export_data(self) -> Dict[str, Any]:
        """
        Request export of all user data.

        Returns:
            Export request confirmation with expected completion time
        """
        return self._post(f"{Endpoints.USER_ME}/export")

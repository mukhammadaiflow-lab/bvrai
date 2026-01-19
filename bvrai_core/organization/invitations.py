"""
Invitation System

Organization member invitations and team assignments.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from .auth import generate_token, hash_token
from .base import (
    Invitation,
    InvitationError,
    InvitationStatus,
    InvitationStore,
    MemberRole,
    Organization,
    OrganizationMember,
    User,
    UserStatus,
)
from .models import (
    InMemoryInvitationStore,
    InMemoryMemberStore,
    InMemoryOrganizationStore,
    InMemoryUserStore,
)


logger = logging.getLogger(__name__)


@dataclass
class InvitationConfig:
    """Invitation configuration."""

    expiry_days: int = 7
    max_pending_per_org: int = 100
    allow_resend: bool = True
    resend_cooldown_hours: int = 1


class InvitationManager:
    """
    Manager for organization invitations.

    Handles sending, accepting, and revoking invitations.
    """

    def __init__(
        self,
        invitation_store: InMemoryInvitationStore,
        user_store: InMemoryUserStore,
        org_store: InMemoryOrganizationStore,
        member_store: InMemoryMemberStore,
        config: Optional[InvitationConfig] = None,
    ):
        """Initialize invitation manager."""
        self._invitations = invitation_store
        self._users = user_store
        self._orgs = org_store
        self._members = member_store
        self._config = config or InvitationConfig()

        # Notification callbacks
        self._notification_callbacks: List[Callable] = []

    def add_notification_callback(self, callback: Callable) -> None:
        """Add callback for invitation notifications."""
        self._notification_callbacks.append(callback)

    async def _notify(
        self,
        event_type: str,
        invitation: Invitation,
        organization: Organization,
        inviter: Optional[User] = None,
    ) -> None:
        """Send notification about invitation."""
        for callback in self._notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, invitation, organization, inviter)
                else:
                    callback(event_type, invitation, organization, inviter)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")

    async def create_invitation(
        self,
        organization_id: str,
        email: str,
        role: MemberRole,
        invited_by_user_id: str,
        team_id: Optional[str] = None,
        custom_message: Optional[str] = None,
    ) -> Invitation:
        """
        Create an invitation to join an organization.

        Raises:
            InvitationError: If invitation cannot be created
        """
        import uuid

        # Get organization
        org = await self._orgs.get(organization_id)
        if not org:
            raise InvitationError(f"Organization not found: {organization_id}")

        # Check if user is already a member
        user = await self._users.get_by_email(email)
        if user:
            existing_member = await self._members.get_by_user_and_org(user.id, organization_id)
            if existing_member:
                raise InvitationError(f"User {email} is already a member")

        # Check for existing pending invitation
        pending = await self._invitations.list_for_organization(
            organization_id,
            status=InvitationStatus.PENDING,
        )

        existing = [i for i in pending if i.email.lower() == email.lower()]
        if existing:
            # Check if we can resend
            if self._config.allow_resend:
                last_invite = max(existing, key=lambda x: x.created_at)
                cooldown = timedelta(hours=self._config.resend_cooldown_hours)
                if datetime.utcnow() - last_invite.created_at < cooldown:
                    raise InvitationError(
                        f"Invitation already sent. Please wait before resending."
                    )
                # Revoke old invitation
                await self.revoke_invitation(last_invite.id)
            else:
                raise InvitationError(f"Pending invitation already exists for {email}")

        # Check max pending
        if len(pending) >= self._config.max_pending_per_org:
            raise InvitationError("Maximum pending invitations reached")

        # Generate token
        token = generate_token()

        invitation = Invitation(
            id=f"inv_{uuid.uuid4().hex[:16]}",
            organization_id=organization_id,
            email=email,
            role=role,
            status=InvitationStatus.PENDING,
            invited_by_user_id=invited_by_user_id,
            token=token,
            expires_at=datetime.utcnow() + timedelta(days=self._config.expiry_days),
            team_id=team_id,
            metadata={"custom_message": custom_message} if custom_message else {},
        )

        await self._invitations.create(invitation)

        # Send notification
        inviter = await self._users.get(invited_by_user_id)
        await self._notify("invitation_sent", invitation, org, inviter)

        logger.info(f"Created invitation for {email} to {org.name}")

        return invitation

    async def get_invitation(self, invitation_id: str) -> Optional[Invitation]:
        """Get invitation by ID."""
        return await self._invitations.get(invitation_id)

    async def get_invitation_by_token(self, token: str) -> Optional[Invitation]:
        """Get invitation by token."""
        return await self._invitations.get_by_token(token)

    async def accept_invitation(
        self,
        token: str,
        user: Optional[User] = None,
        create_user: bool = False,
        user_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[OrganizationMember, Organization]:
        """
        Accept an invitation.

        Args:
            token: Invitation token
            user: Existing user (if logged in)
            create_user: Whether to create user if not exists
            user_data: Data for creating new user

        Returns:
            Tuple of (membership, organization)

        Raises:
            InvitationError: If acceptance fails
        """
        import uuid

        invitation = await self._invitations.get_by_token(token)
        if not invitation:
            raise InvitationError("Invalid invitation token")

        if not invitation.is_valid():
            if invitation.status == InvitationStatus.EXPIRED:
                raise InvitationError("Invitation has expired")
            elif invitation.status == InvitationStatus.REVOKED:
                raise InvitationError("Invitation has been revoked")
            elif invitation.status == InvitationStatus.ACCEPTED:
                raise InvitationError("Invitation has already been accepted")
            raise InvitationError("Invitation is no longer valid")

        # Get organization
        org = await self._orgs.get(invitation.organization_id)
        if not org or not org.is_active():
            raise InvitationError("Organization is not available")

        # Determine user
        if user:
            # Verify email matches
            if user.email.lower() != invitation.email.lower():
                raise InvitationError("Invitation was sent to a different email")
        else:
            # Try to find existing user
            user = await self._users.get_by_email(invitation.email)

            if not user:
                if not create_user:
                    raise InvitationError("Please create an account first")

                # Create user
                user = User(
                    id=f"user_{uuid.uuid4().hex[:16]}",
                    email=invitation.email,
                    status=UserStatus.ACTIVE,
                    email_verified=True,
                    email_verified_at=datetime.utcnow(),
                    **(user_data or {}),
                )
                await self._users.create(user)

        # Check if already a member
        existing = await self._members.get_by_user_and_org(user.id, org.id)
        if existing:
            raise InvitationError("Already a member of this organization")

        # Create membership
        member = OrganizationMember(
            id=f"member_{uuid.uuid4().hex[:16]}",
            organization_id=org.id,
            user_id=user.id,
            role=invitation.role,
            accepted=True,
            accepted_at=datetime.utcnow(),
        )

        await self._members.create(member)

        # Update invitation
        invitation.status = InvitationStatus.ACCEPTED
        invitation.accepted_at = datetime.utcnow()
        await self._invitations.update(invitation)

        # Send notification
        await self._notify("invitation_accepted", invitation, org, user)

        logger.info(f"User {user.email} accepted invitation to {org.name}")

        return (member, org)

    async def revoke_invitation(self, invitation_id: str) -> bool:
        """Revoke an invitation."""
        invitation = await self._invitations.get(invitation_id)
        if not invitation:
            return False

        if invitation.status != InvitationStatus.PENDING:
            return False

        invitation.status = InvitationStatus.REVOKED
        invitation.revoked_at = datetime.utcnow()
        await self._invitations.update(invitation)

        return True

    async def resend_invitation(
        self,
        invitation_id: str,
    ) -> Invitation:
        """
        Resend an invitation (creates new token).

        Raises:
            InvitationError: If resend fails
        """
        old_invitation = await self._invitations.get(invitation_id)
        if not old_invitation:
            raise InvitationError("Invitation not found")

        if old_invitation.status != InvitationStatus.PENDING:
            raise InvitationError("Can only resend pending invitations")

        # Check cooldown
        cooldown = timedelta(hours=self._config.resend_cooldown_hours)
        if datetime.utcnow() - old_invitation.created_at < cooldown:
            raise InvitationError("Please wait before resending")

        # Revoke old and create new
        await self.revoke_invitation(invitation_id)

        return await self.create_invitation(
            organization_id=old_invitation.organization_id,
            email=old_invitation.email,
            role=old_invitation.role,
            invited_by_user_id=old_invitation.invited_by_user_id,
            team_id=old_invitation.team_id,
            custom_message=old_invitation.metadata.get("custom_message"),
        )

    async def list_organization_invitations(
        self,
        organization_id: str,
        status: Optional[InvitationStatus] = None,
    ) -> List[Invitation]:
        """List invitations for organization."""
        return await self._invitations.list_for_organization(organization_id, status)

    async def list_user_invitations(
        self,
        email: str,
        status: Optional[InvitationStatus] = None,
    ) -> List[Invitation]:
        """List invitations for a user email."""
        invitations = await self._invitations.list_for_email(email)

        if status:
            invitations = [i for i in invitations if i.status == status]

        return invitations

    async def expire_old_invitations(self) -> int:
        """Expire invitations past their expiry date."""
        count = 0

        # This is inefficient for large datasets - in production
        # would use database query
        for org_invites in self._invitations._by_org.values():
            for inv_id in list(org_invites):
                invitation = await self._invitations.get(inv_id)
                if invitation and invitation.status == InvitationStatus.PENDING:
                    if datetime.utcnow() > invitation.expires_at:
                        invitation.status = InvitationStatus.EXPIRED
                        await self._invitations.update(invitation)
                        count += 1

        return count

    async def get_invitation_stats(
        self,
        organization_id: str,
    ) -> Dict[str, int]:
        """Get invitation statistics for organization."""
        invitations = await self._invitations.list_for_organization(organization_id)

        stats = {
            "total": len(invitations),
            "pending": 0,
            "accepted": 0,
            "expired": 0,
            "revoked": 0,
        }

        for inv in invitations:
            status_key = inv.status.value
            if status_key in stats:
                stats[status_key] += 1

        return stats


class BulkInvitationManager:
    """
    Manager for bulk invitation operations.

    Handles inviting multiple users at once.
    """

    def __init__(self, invitation_manager: InvitationManager):
        """Initialize bulk manager."""
        self._manager = invitation_manager

    async def send_bulk_invitations(
        self,
        organization_id: str,
        invitations: List[Dict[str, Any]],
        invited_by_user_id: str,
        default_role: MemberRole = MemberRole.MEMBER,
    ) -> Dict[str, Any]:
        """
        Send multiple invitations.

        Args:
            organization_id: Organization to invite to
            invitations: List of {"email": str, "role": Optional[MemberRole]}
            invited_by_user_id: User sending invitations
            default_role: Default role if not specified

        Returns:
            Summary of results
        """
        results = {
            "successful": [],
            "failed": [],
            "skipped": [],
        }

        for inv_data in invitations:
            email = inv_data.get("email")
            if not email:
                results["skipped"].append({"email": None, "reason": "No email provided"})
                continue

            role = inv_data.get("role", default_role)
            team_id = inv_data.get("team_id")

            try:
                invitation = await self._manager.create_invitation(
                    organization_id=organization_id,
                    email=email,
                    role=role,
                    invited_by_user_id=invited_by_user_id,
                    team_id=team_id,
                )
                results["successful"].append({
                    "email": email,
                    "invitation_id": invitation.id,
                })
            except InvitationError as e:
                results["failed"].append({
                    "email": email,
                    "error": str(e),
                })
            except Exception as e:
                logger.error(f"Error inviting {email}: {e}")
                results["failed"].append({
                    "email": email,
                    "error": "Internal error",
                })

        results["summary"] = {
            "total": len(invitations),
            "successful": len(results["successful"]),
            "failed": len(results["failed"]),
            "skipped": len(results["skipped"]),
        }

        return results

    async def import_from_csv(
        self,
        organization_id: str,
        csv_content: str,
        invited_by_user_id: str,
        default_role: MemberRole = MemberRole.MEMBER,
    ) -> Dict[str, Any]:
        """
        Import invitations from CSV.

        CSV format: email,role (header optional)
        """
        import csv
        from io import StringIO

        invitations = []

        reader = csv.reader(StringIO(csv_content))
        header = None

        for row in reader:
            if not row:
                continue

            # Check if this is a header row
            if row[0].lower() == "email":
                header = row
                continue

            email = row[0].strip() if row else None
            role_str = row[1].strip() if len(row) > 1 else None

            if email:
                role = default_role
                if role_str:
                    try:
                        role = MemberRole(role_str.lower())
                    except ValueError:
                        pass  # Use default

                invitations.append({"email": email, "role": role})

        return await self.send_bulk_invitations(
            organization_id=organization_id,
            invitations=invitations,
            invited_by_user_id=invited_by_user_id,
            default_role=default_role,
        )

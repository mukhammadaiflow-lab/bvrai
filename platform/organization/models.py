"""
Organization Models

In-memory implementations of storage interfaces for development and testing.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

from .base import (
    APIKey,
    AuditLogEntry,
    Invitation,
    InvitationStatus,
    InvitationStore,
    MemberRole,
    MemberStore,
    Organization,
    OrganizationMember,
    OrganizationStore,
    Session,
    Team,
    User,
    UserStore,
)


logger = logging.getLogger(__name__)


class InMemoryUserStore(UserStore):
    """In-memory user store implementation."""

    def __init__(self):
        """Initialize store."""
        self._users: Dict[str, User] = {}
        self._by_email: Dict[str, str] = {}

    async def create(self, user: User) -> None:
        """Create user."""
        self._users[user.id] = user
        self._by_email[user.email.lower()] = user.id

    async def get(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        user_id = self._by_email.get(email.lower())
        if user_id:
            return self._users.get(user_id)
        return None

    async def update(self, user: User) -> None:
        """Update user."""
        old_user = self._users.get(user.id)
        if old_user and old_user.email.lower() != user.email.lower():
            del self._by_email[old_user.email.lower()]
            self._by_email[user.email.lower()] = user.id
        self._users[user.id] = user

    async def delete(self, user_id: str) -> None:
        """Delete user."""
        user = self._users.get(user_id)
        if user:
            self._by_email.pop(user.email.lower(), None)
            del self._users[user_id]

    async def list_all(self) -> List[User]:
        """List all users."""
        return list(self._users.values())


class InMemoryOrganizationStore(OrganizationStore):
    """In-memory organization store implementation."""

    def __init__(self, member_store: "InMemoryMemberStore"):
        """Initialize store."""
        self._organizations: Dict[str, Organization] = {}
        self._by_slug: Dict[str, str] = {}
        self._member_store = member_store

    async def create(self, org: Organization) -> None:
        """Create organization."""
        self._organizations[org.id] = org
        self._by_slug[org.slug.lower()] = org.id

    async def get(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID."""
        return self._organizations.get(org_id)

    async def get_by_slug(self, slug: str) -> Optional[Organization]:
        """Get organization by slug."""
        org_id = self._by_slug.get(slug.lower())
        if org_id:
            return self._organizations.get(org_id)
        return None

    async def update(self, org: Organization) -> None:
        """Update organization."""
        old_org = self._organizations.get(org.id)
        if old_org and old_org.slug.lower() != org.slug.lower():
            del self._by_slug[old_org.slug.lower()]
            self._by_slug[org.slug.lower()] = org.id
        self._organizations[org.id] = org

    async def delete(self, org_id: str) -> None:
        """Delete organization."""
        org = self._organizations.get(org_id)
        if org:
            self._by_slug.pop(org.slug.lower(), None)
            del self._organizations[org_id]

    async def list_for_user(self, user_id: str) -> List[Organization]:
        """List organizations for user."""
        members = await self._member_store.list_for_user(user_id)
        org_ids = {m.organization_id for m in members}
        return [
            org for org in self._organizations.values()
            if org.id in org_ids
        ]

    async def list_all(self) -> List[Organization]:
        """List all organizations."""
        return list(self._organizations.values())


class InMemoryMemberStore(MemberStore):
    """In-memory membership store implementation."""

    def __init__(self):
        """Initialize store."""
        self._members: Dict[str, OrganizationMember] = {}
        self._by_org: Dict[str, Set[str]] = defaultdict(set)
        self._by_user: Dict[str, Set[str]] = defaultdict(set)
        self._by_user_org: Dict[str, str] = {}  # "user_id:org_id" -> member_id

    async def create(self, member: OrganizationMember) -> None:
        """Create membership."""
        self._members[member.id] = member
        self._by_org[member.organization_id].add(member.id)
        self._by_user[member.user_id].add(member.id)
        self._by_user_org[f"{member.user_id}:{member.organization_id}"] = member.id

    async def get(self, member_id: str) -> Optional[OrganizationMember]:
        """Get membership by ID."""
        return self._members.get(member_id)

    async def get_by_user_and_org(
        self,
        user_id: str,
        org_id: str,
    ) -> Optional[OrganizationMember]:
        """Get membership by user and organization."""
        member_id = self._by_user_org.get(f"{user_id}:{org_id}")
        if member_id:
            return self._members.get(member_id)
        return None

    async def update(self, member: OrganizationMember) -> None:
        """Update membership."""
        self._members[member.id] = member

    async def delete(self, member_id: str) -> None:
        """Delete membership."""
        member = self._members.get(member_id)
        if member:
            self._by_org[member.organization_id].discard(member_id)
            self._by_user[member.user_id].discard(member_id)
            self._by_user_org.pop(f"{member.user_id}:{member.organization_id}", None)
            del self._members[member_id]

    async def list_for_organization(self, org_id: str) -> List[OrganizationMember]:
        """List members for organization."""
        member_ids = self._by_org.get(org_id, set())
        return [self._members[mid] for mid in member_ids if mid in self._members]

    async def list_for_user(self, user_id: str) -> List[OrganizationMember]:
        """List memberships for user."""
        member_ids = self._by_user.get(user_id, set())
        return [self._members[mid] for mid in member_ids if mid in self._members]


class InMemoryInvitationStore(InvitationStore):
    """In-memory invitation store implementation."""

    def __init__(self):
        """Initialize store."""
        self._invitations: Dict[str, Invitation] = {}
        self._by_token: Dict[str, str] = {}
        self._by_org: Dict[str, Set[str]] = defaultdict(set)
        self._by_email: Dict[str, Set[str]] = defaultdict(set)

    async def create(self, invitation: Invitation) -> None:
        """Create invitation."""
        self._invitations[invitation.id] = invitation
        self._by_token[invitation.token] = invitation.id
        self._by_org[invitation.organization_id].add(invitation.id)
        self._by_email[invitation.email.lower()].add(invitation.id)

    async def get(self, invitation_id: str) -> Optional[Invitation]:
        """Get invitation by ID."""
        return self._invitations.get(invitation_id)

    async def get_by_token(self, token: str) -> Optional[Invitation]:
        """Get invitation by token."""
        invitation_id = self._by_token.get(token)
        if invitation_id:
            return self._invitations.get(invitation_id)
        return None

    async def update(self, invitation: Invitation) -> None:
        """Update invitation."""
        self._invitations[invitation.id] = invitation

    async def list_for_organization(
        self,
        org_id: str,
        status: Optional[InvitationStatus] = None,
    ) -> List[Invitation]:
        """List invitations for organization."""
        invitation_ids = self._by_org.get(org_id, set())
        invitations = [self._invitations[iid] for iid in invitation_ids if iid in self._invitations]

        if status:
            invitations = [i for i in invitations if i.status == status]

        return invitations

    async def list_for_email(self, email: str) -> List[Invitation]:
        """List invitations for email."""
        invitation_ids = self._by_email.get(email.lower(), set())
        return [self._invitations[iid] for iid in invitation_ids if iid in self._invitations]


class InMemoryTeamStore:
    """In-memory team store implementation."""

    def __init__(self):
        """Initialize store."""
        self._teams: Dict[str, Team] = {}
        self._by_org: Dict[str, Set[str]] = defaultdict(set)

    async def create(self, team: Team) -> None:
        """Create team."""
        self._teams[team.id] = team
        self._by_org[team.organization_id].add(team.id)

    async def get(self, team_id: str) -> Optional[Team]:
        """Get team by ID."""
        return self._teams.get(team_id)

    async def update(self, team: Team) -> None:
        """Update team."""
        self._teams[team.id] = team

    async def delete(self, team_id: str) -> None:
        """Delete team."""
        team = self._teams.get(team_id)
        if team:
            self._by_org[team.organization_id].discard(team_id)
            del self._teams[team_id]

    async def list_for_organization(self, org_id: str) -> List[Team]:
        """List teams for organization."""
        team_ids = self._by_org.get(org_id, set())
        return [self._teams[tid] for tid in team_ids if tid in self._teams]

    async def add_member(self, team_id: str, user_id: str) -> bool:
        """Add member to team."""
        team = self._teams.get(team_id)
        if team:
            team.member_ids.add(user_id)
            return True
        return False

    async def remove_member(self, team_id: str, user_id: str) -> bool:
        """Remove member from team."""
        team = self._teams.get(team_id)
        if team:
            team.member_ids.discard(user_id)
            return True
        return False


class InMemoryAPIKeyStore:
    """In-memory API key store implementation."""

    def __init__(self):
        """Initialize store."""
        self._keys: Dict[str, APIKey] = {}
        self._by_org: Dict[str, Set[str]] = defaultdict(set)
        self._by_hash: Dict[str, str] = {}

    async def create(self, api_key: APIKey) -> None:
        """Create API key."""
        self._keys[api_key.id] = api_key
        self._by_org[api_key.organization_id].add(api_key.id)
        self._by_hash[api_key.key_hash] = api_key.id

    async def get(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        return self._keys.get(key_id)

    async def get_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash."""
        key_id = self._by_hash.get(key_hash)
        if key_id:
            return self._keys.get(key_id)
        return None

    async def update(self, api_key: APIKey) -> None:
        """Update API key."""
        self._keys[api_key.id] = api_key

    async def delete(self, key_id: str) -> None:
        """Delete API key."""
        api_key = self._keys.get(key_id)
        if api_key:
            self._by_org[api_key.organization_id].discard(key_id)
            self._by_hash.pop(api_key.key_hash, None)
            del self._keys[key_id]

    async def list_for_organization(self, org_id: str) -> List[APIKey]:
        """List API keys for organization."""
        key_ids = self._by_org.get(org_id, set())
        return [self._keys[kid] for kid in key_ids if kid in self._keys]


class InMemorySessionStore:
    """In-memory session store implementation."""

    def __init__(self):
        """Initialize store."""
        self._sessions: Dict[str, Session] = {}
        self._by_user: Dict[str, Set[str]] = defaultdict(set)
        self._by_token: Dict[str, str] = {}

    async def create(self, session: Session) -> None:
        """Create session."""
        self._sessions[session.id] = session
        self._by_user[session.user_id].add(session.id)
        self._by_token[session.token_hash] = session.id

    async def get(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    async def get_by_token(self, token_hash: str) -> Optional[Session]:
        """Get session by token hash."""
        session_id = self._by_token.get(token_hash)
        if session_id:
            return self._sessions.get(session_id)
        return None

    async def update(self, session: Session) -> None:
        """Update session."""
        self._sessions[session.id] = session

    async def delete(self, session_id: str) -> None:
        """Delete session."""
        session = self._sessions.get(session_id)
        if session:
            self._by_user[session.user_id].discard(session_id)
            self._by_token.pop(session.token_hash, None)
            del self._sessions[session_id]

    async def list_for_user(self, user_id: str) -> List[Session]:
        """List sessions for user."""
        session_ids = self._by_user.get(user_id, set())
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]

    async def revoke_all_for_user(self, user_id: str) -> int:
        """Revoke all sessions for user."""
        from datetime import datetime

        session_ids = self._by_user.get(user_id, set())
        count = 0
        for session_id in list(session_ids):
            session = self._sessions.get(session_id)
            if session and session.active:
                session.active = False
                session.revoked_at = datetime.utcnow()
                count += 1
        return count


class InMemoryAuditLogStore:
    """In-memory audit log store implementation."""

    def __init__(self, max_entries: int = 10000):
        """Initialize store."""
        self._entries: Dict[str, AuditLogEntry] = {}
        self._by_org: Dict[str, List[str]] = defaultdict(list)
        self._max_entries = max_entries

    async def create(self, entry: AuditLogEntry) -> None:
        """Create audit log entry."""
        self._entries[entry.id] = entry
        self._by_org[entry.organization_id].append(entry.id)

        # Trim if over limit
        if len(self._entries) > self._max_entries:
            oldest = min(self._entries.values(), key=lambda e: e.timestamp)
            await self.delete(oldest.id)

    async def get(self, entry_id: str) -> Optional[AuditLogEntry]:
        """Get audit log entry by ID."""
        return self._entries.get(entry_id)

    async def delete(self, entry_id: str) -> None:
        """Delete audit log entry."""
        entry = self._entries.get(entry_id)
        if entry:
            self._by_org[entry.organization_id].remove(entry_id)
            del self._entries[entry_id]

    async def list_for_organization(
        self,
        org_id: str,
        limit: int = 100,
        offset: int = 0,
        resource_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """List audit log entries for organization."""
        entry_ids = self._by_org.get(org_id, [])
        entries = [self._entries[eid] for eid in entry_ids if eid in self._entries]

        # Sort by timestamp descending
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        # Filter
        if resource_type:
            entries = [e for e in entries if e.resource_type == resource_type]
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]

        return entries[offset:offset + limit]

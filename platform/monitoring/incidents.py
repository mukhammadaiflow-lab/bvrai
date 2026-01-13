"""
Incident Management System

Provides comprehensive incident tracking, management, and reporting
for system outages and degradations.

Features:
- Incident creation and lifecycle management
- Timeline and update tracking
- Severity classification (SEV1-SEV5)
- Responder assignment
- Postmortem tracking
- SLA monitoring
"""

import asyncio
import json
import logging
import secrets
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class IncidentSeverity(str, Enum):
    """Incident severity levels."""
    SEV1 = "sev1"  # Critical - immediate action required
    SEV2 = "sev2"  # High - urgent attention needed
    SEV3 = "sev3"  # Medium - needs attention soon
    SEV4 = "sev4"  # Low - minor issue
    SEV5 = "sev5"  # Info - informational


class IncidentStatus(str, Enum):
    """Incident status."""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    IMPLEMENTING = "implementing"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    POSTMORTEM = "postmortem"
    CLOSED = "closed"


class UpdateType(str, Enum):
    """Types of incident updates."""
    STATUS_CHANGE = "status_change"
    MESSAGE = "message"
    RESPONDER_ADDED = "responder_added"
    RESPONDER_REMOVED = "responder_removed"
    SEVERITY_CHANGE = "severity_change"
    IMPACT_UPDATED = "impact_updated"
    ACTION_ITEM = "action_item"


@dataclass
class IncidentUpdate:
    """An update to an incident."""
    id: str
    incident_id: str
    update_type: UpdateType
    message: str
    author: str
    status: Optional[IncidentStatus] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_public: bool = True  # Visible on status page

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "update_type": self.update_type.value,
            "message": self.message,
            "author": self.author,
            "status": self.status.value if self.status else None,
            "created_at": self.created_at.isoformat(),
            "is_public": self.is_public,
        }


@dataclass
class ActionItem:
    """Action item for incident resolution."""
    id: str
    incident_id: str
    description: str
    assignee: Optional[str] = None
    completed: bool = False
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    due_at: Optional[datetime] = None
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "description": self.description,
            "assignee": self.assignee,
            "completed": self.completed,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "due_at": self.due_at.isoformat() if self.due_at else None,
            "priority": self.priority,
        }


@dataclass
class Postmortem:
    """Postmortem/post-incident review."""
    id: str
    incident_id: str
    title: str
    summary: str = ""
    timeline: str = ""
    root_cause: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    attendees: List[str] = field(default_factory=list)
    status: str = "draft"  # draft, review, published
    created_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    document_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "title": self.title,
            "summary": self.summary,
            "root_cause": self.root_cause,
            "contributing_factors": self.contributing_factors,
            "lessons_learned": self.lessons_learned,
            "action_items": self.action_items,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "document_url": self.document_url,
        }


@dataclass
class Incident:
    """An incident record."""
    id: str
    organization_id: Optional[str]  # None for platform-wide incidents
    title: str
    severity: IncidentSeverity
    status: IncidentStatus

    # Description
    description: str = ""
    impact: str = ""
    customer_impact: str = ""
    root_cause: str = ""
    resolution: str = ""

    # Classification
    service: str = ""
    components: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    detected_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    identified_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # SLA tracking
    time_to_acknowledge_seconds: Optional[int] = None
    time_to_mitigate_seconds: Optional[int] = None
    time_to_resolve_seconds: Optional[int] = None
    sla_breached: bool = False

    # Responders
    commander: Optional[str] = None
    communications_lead: Optional[str] = None
    responders: List[str] = field(default_factory=list)
    on_call_notified: bool = False

    # Related items
    alert_ids: List[str] = field(default_factory=list)
    related_incidents: List[str] = field(default_factory=list)
    external_ticket_url: Optional[str] = None

    # Updates and actions
    updates: List[IncidentUpdate] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)
    postmortem_id: Optional[str] = None

    # Notification tracking
    notification_sent: bool = False
    status_page_updated: bool = False

    @property
    def duration_seconds(self) -> Optional[int]:
        """Get incident duration in seconds."""
        start = self.started_at or self.detected_at or self.created_at
        end = self.resolved_at or datetime.utcnow()
        return int((end - start).total_seconds())

    @property
    def is_active(self) -> bool:
        """Check if incident is still active."""
        return self.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]

    @property
    def time_since_last_update_seconds(self) -> int:
        """Get seconds since last update."""
        if self.updates:
            last_update = max(u.created_at for u in self.updates)
        else:
            last_update = self.created_at
        return int((datetime.utcnow() - last_update).total_seconds())

    def to_dict(self, include_updates: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "organization_id": self.organization_id,
            "title": self.title,
            "severity": self.severity.value,
            "status": self.status.value,
            "description": self.description,
            "impact": self.impact,
            "service": self.service,
            "components": self.components,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "duration_seconds": self.duration_seconds,
            "commander": self.commander,
            "responders": self.responders,
            "updates_count": len(self.updates),
            "action_items_count": len(self.action_items),
            "is_active": self.is_active,
        }

        if include_updates:
            result["updates"] = [u.to_dict() for u in self.updates]
            result["action_items"] = [a.to_dict() for a in self.action_items]

        return result


@dataclass
class IncidentConfig:
    """Configuration for incident management."""
    # SLA targets (in seconds)
    sla_acknowledge_sev1: int = 300  # 5 minutes
    sla_acknowledge_sev2: int = 900  # 15 minutes
    sla_acknowledge_sev3: int = 3600  # 1 hour
    sla_mitigate_sev1: int = 1800  # 30 minutes
    sla_mitigate_sev2: int = 7200  # 2 hours
    sla_mitigate_sev3: int = 28800  # 8 hours

    # Escalation rules
    escalate_after_no_update_seconds: int = 1800  # 30 minutes
    auto_escalate_severity: bool = True

    # Notifications
    notify_on_create: bool = True
    notify_on_status_change: bool = True
    notify_on_resolution: bool = True

    # Postmortem
    require_postmortem_sev1: bool = True
    require_postmortem_sev2: bool = True
    postmortem_deadline_days: int = 5


class IncidentManager:
    """
    Manages incidents throughout their lifecycle.

    Features:
    - Incident CRUD operations
    - Status transitions
    - Update tracking
    - Responder management
    - SLA monitoring
    - Escalation
    """

    def __init__(self, config: Optional[IncidentConfig] = None):
        """
        Initialize incident manager.

        Args:
            config: Incident management configuration
        """
        self.config = config or IncidentConfig()

        # Storage
        self._incidents: Dict[str, Incident] = {}
        self._postmortems: Dict[str, Postmortem] = {}
        self._lock = asyncio.Lock()

        # Event handlers
        self._on_created: List[Callable] = []
        self._on_updated: List[Callable] = []
        self._on_resolved: List[Callable] = []
        self._on_escalation: List[Callable] = []

        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    # =========================================================================
    # Incident CRUD
    # =========================================================================

    async def create_incident(
        self,
        title: str,
        severity: IncidentSeverity,
        organization_id: Optional[str] = None,
        description: str = "",
        impact: str = "",
        service: str = "",
        components: Optional[List[str]] = None,
        commander: Optional[str] = None,
        alert_ids: Optional[List[str]] = None,
    ) -> Incident:
        """
        Create a new incident.

        Args:
            title: Incident title
            severity: Incident severity
            organization_id: Organization ID (None for platform-wide)
            description: Incident description
            impact: Impact description
            service: Affected service
            components: Affected components
            commander: Incident commander
            alert_ids: Related alert IDs

        Returns:
            Created incident
        """
        incident_id = f"inc_{secrets.token_hex(8)}"
        now = datetime.utcnow()

        incident = Incident(
            id=incident_id,
            organization_id=organization_id,
            title=title,
            severity=severity,
            status=IncidentStatus.DETECTED,
            description=description,
            impact=impact,
            service=service,
            components=components or [],
            commander=commander,
            alert_ids=alert_ids or [],
            detected_at=now,
            created_at=now,
        )

        # Add initial update
        incident.updates.append(IncidentUpdate(
            id=f"upd_{secrets.token_hex(6)}",
            incident_id=incident_id,
            update_type=UpdateType.STATUS_CHANGE,
            message=f"Incident created: {title}",
            author="system",
            status=IncidentStatus.DETECTED,
        ))

        async with self._lock:
            self._incidents[incident_id] = incident

        # Emit created event
        for handler in self._on_created:
            try:
                await handler(incident)
            except Exception as e:
                logger.error(f"Error in incident created handler: {e}")

        logger.info(f"Created incident: {incident_id} - {title} (severity: {severity.value})")
        return incident

    async def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get an incident by ID."""
        async with self._lock:
            return self._incidents.get(incident_id)

    async def list_incidents(
        self,
        organization_id: Optional[str] = None,
        status: Optional[IncidentStatus] = None,
        severity: Optional[IncidentSeverity] = None,
        active_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Incident]:
        """
        List incidents with filters.

        Args:
            organization_id: Filter by organization
            status: Filter by status
            severity: Filter by severity
            active_only: Only return active incidents
            limit: Maximum number to return
            offset: Offset for pagination

        Returns:
            List of matching incidents
        """
        async with self._lock:
            incidents = list(self._incidents.values())

        # Apply filters
        if organization_id is not None:
            incidents = [i for i in incidents if i.organization_id == organization_id]
        if status:
            incidents = [i for i in incidents if i.status == status]
        if severity:
            incidents = [i for i in incidents if i.severity == severity]
        if active_only:
            incidents = [i for i in incidents if i.is_active]

        # Sort by creation time (newest first)
        incidents.sort(key=lambda i: i.created_at, reverse=True)

        # Paginate
        return incidents[offset:offset + limit]

    async def update_incident(
        self,
        incident_id: str,
        updates: Dict[str, Any],
        author: str,
        message: str = "",
    ) -> Optional[Incident]:
        """
        Update an incident.

        Args:
            incident_id: Incident ID
            updates: Fields to update
            author: Author of the update
            message: Update message

        Returns:
            Updated incident or None
        """
        async with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return None

            # Track status change
            old_status = incident.status

            # Apply updates
            for key, value in updates.items():
                if hasattr(incident, key):
                    setattr(incident, key, value)

            # Record update
            update_type = UpdateType.MESSAGE
            if "status" in updates and updates["status"] != old_status:
                update_type = UpdateType.STATUS_CHANGE
            elif "severity" in updates:
                update_type = UpdateType.SEVERITY_CHANGE

            incident.updates.append(IncidentUpdate(
                id=f"upd_{secrets.token_hex(6)}",
                incident_id=incident_id,
                update_type=update_type,
                message=message or f"Incident updated by {author}",
                author=author,
                status=updates.get("status") if isinstance(updates.get("status"), IncidentStatus) else None,
            ))

        # Emit updated event
        for handler in self._on_updated:
            try:
                await handler(incident, updates)
            except Exception as e:
                logger.error(f"Error in incident updated handler: {e}")

        logger.info(f"Updated incident: {incident_id}")
        return incident

    async def delete_incident(self, incident_id: str) -> bool:
        """
        Delete an incident.

        Args:
            incident_id: Incident ID

        Returns:
            True if deleted
        """
        async with self._lock:
            if incident_id in self._incidents:
                del self._incidents[incident_id]
                return True
        return False

    # =========================================================================
    # Status Management
    # =========================================================================

    async def acknowledge_incident(
        self,
        incident_id: str,
        commander: str,
        message: str = "",
    ) -> Optional[Incident]:
        """
        Acknowledge an incident and assign a commander.

        Args:
            incident_id: Incident ID
            commander: Incident commander
            message: Acknowledgment message

        Returns:
            Updated incident or None
        """
        incident = await self.get_incident(incident_id)
        if not incident:
            return None

        # Calculate time to acknowledge
        start_time = incident.detected_at or incident.created_at
        incident.time_to_acknowledge_seconds = int(
            (datetime.utcnow() - start_time).total_seconds()
        )

        # Check SLA
        sla_targets = {
            IncidentSeverity.SEV1: self.config.sla_acknowledge_sev1,
            IncidentSeverity.SEV2: self.config.sla_acknowledge_sev2,
            IncidentSeverity.SEV3: self.config.sla_acknowledge_sev3,
        }
        target = sla_targets.get(incident.severity, float('inf'))
        if incident.time_to_acknowledge_seconds > target:
            incident.sla_breached = True

        return await self.update_incident(
            incident_id,
            {
                "status": IncidentStatus.INVESTIGATING,
                "commander": commander,
            },
            author=commander,
            message=message or f"Incident acknowledged by {commander}",
        )

    async def update_status(
        self,
        incident_id: str,
        status: IncidentStatus,
        author: str,
        message: str,
        is_public: bool = True,
    ) -> Optional[Incident]:
        """
        Update incident status.

        Args:
            incident_id: Incident ID
            status: New status
            author: Author of the update
            message: Status update message
            is_public: Whether update is visible on status page

        Returns:
            Updated incident or None
        """
        async with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return None

            old_status = incident.status
            incident.status = status

            # Track timing milestones
            now = datetime.utcnow()
            if status == IncidentStatus.IDENTIFIED and not incident.identified_at:
                incident.identified_at = now
            elif status == IncidentStatus.RESOLVED and not incident.resolved_at:
                incident.resolved_at = now
                start_time = incident.started_at or incident.detected_at or incident.created_at
                incident.time_to_resolve_seconds = int((now - start_time).total_seconds())
            elif status == IncidentStatus.CLOSED and not incident.closed_at:
                incident.closed_at = now

            # Record update
            incident.updates.append(IncidentUpdate(
                id=f"upd_{secrets.token_hex(6)}",
                incident_id=incident_id,
                update_type=UpdateType.STATUS_CHANGE,
                message=message,
                author=author,
                status=status,
                is_public=is_public,
            ))

        # Emit events
        if status == IncidentStatus.RESOLVED:
            for handler in self._on_resolved:
                try:
                    await handler(incident)
                except Exception as e:
                    logger.error(f"Error in incident resolved handler: {e}")

        for handler in self._on_updated:
            try:
                await handler(incident, {"status": status})
            except Exception as e:
                logger.error(f"Error in incident updated handler: {e}")

        logger.info(f"Incident {incident_id} status changed: {old_status.value} -> {status.value}")
        return incident

    async def resolve_incident(
        self,
        incident_id: str,
        author: str,
        resolution: str,
        root_cause: str = "",
    ) -> Optional[Incident]:
        """
        Resolve an incident.

        Args:
            incident_id: Incident ID
            author: Person resolving
            resolution: Resolution description
            root_cause: Root cause description

        Returns:
            Updated incident or None
        """
        async with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return None

            incident.resolution = resolution
            incident.root_cause = root_cause

        return await self.update_status(
            incident_id,
            IncidentStatus.RESOLVED,
            author,
            f"Incident resolved: {resolution}",
        )

    # =========================================================================
    # Responder Management
    # =========================================================================

    async def add_responder(
        self,
        incident_id: str,
        responder: str,
        role: str = "responder",
        added_by: str = "system",
    ) -> Optional[Incident]:
        """
        Add a responder to an incident.

        Args:
            incident_id: Incident ID
            responder: Responder to add
            role: Responder's role
            added_by: Who added the responder

        Returns:
            Updated incident or None
        """
        async with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return None

            if responder not in incident.responders:
                incident.responders.append(responder)

                incident.updates.append(IncidentUpdate(
                    id=f"upd_{secrets.token_hex(6)}",
                    incident_id=incident_id,
                    update_type=UpdateType.RESPONDER_ADDED,
                    message=f"{responder} joined as {role}",
                    author=added_by,
                    is_public=False,
                ))

        return incident

    async def remove_responder(
        self,
        incident_id: str,
        responder: str,
        removed_by: str,
    ) -> Optional[Incident]:
        """
        Remove a responder from an incident.

        Args:
            incident_id: Incident ID
            responder: Responder to remove
            removed_by: Who removed the responder

        Returns:
            Updated incident or None
        """
        async with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return None

            if responder in incident.responders:
                incident.responders.remove(responder)

                incident.updates.append(IncidentUpdate(
                    id=f"upd_{secrets.token_hex(6)}",
                    incident_id=incident_id,
                    update_type=UpdateType.RESPONDER_REMOVED,
                    message=f"{responder} left the incident",
                    author=removed_by,
                    is_public=False,
                ))

        return incident

    async def assign_commander(
        self,
        incident_id: str,
        commander: str,
        assigned_by: str,
    ) -> Optional[Incident]:
        """
        Assign or reassign incident commander.

        Args:
            incident_id: Incident ID
            commander: New commander
            assigned_by: Who assigned the commander

        Returns:
            Updated incident or None
        """
        return await self.update_incident(
            incident_id,
            {"commander": commander},
            author=assigned_by,
            message=f"Incident commander changed to {commander}",
        )

    # =========================================================================
    # Action Items
    # =========================================================================

    async def add_action_item(
        self,
        incident_id: str,
        description: str,
        author: str,
        assignee: Optional[str] = None,
        due_at: Optional[datetime] = None,
    ) -> Optional[ActionItem]:
        """
        Add an action item to an incident.

        Args:
            incident_id: Incident ID
            description: Action item description
            author: Who created it
            assignee: Who should do it
            due_at: When it's due

        Returns:
            Created action item or None
        """
        async with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return None

            action = ActionItem(
                id=f"act_{secrets.token_hex(6)}",
                incident_id=incident_id,
                description=description,
                assignee=assignee,
                due_at=due_at,
            )

            incident.action_items.append(action)

            incident.updates.append(IncidentUpdate(
                id=f"upd_{secrets.token_hex(6)}",
                incident_id=incident_id,
                update_type=UpdateType.ACTION_ITEM,
                message=f"Action item added: {description}",
                author=author,
                is_public=False,
            ))

        return action

    async def complete_action_item(
        self,
        incident_id: str,
        action_id: str,
        completed_by: str,
    ) -> bool:
        """
        Mark an action item as complete.

        Args:
            incident_id: Incident ID
            action_id: Action item ID
            completed_by: Who completed it

        Returns:
            True if completed
        """
        async with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return False

            for action in incident.action_items:
                if action.id == action_id:
                    action.completed = True
                    action.completed_at = datetime.utcnow()

                    incident.updates.append(IncidentUpdate(
                        id=f"upd_{secrets.token_hex(6)}",
                        incident_id=incident_id,
                        update_type=UpdateType.ACTION_ITEM,
                        message=f"Action item completed: {action.description}",
                        author=completed_by,
                        is_public=False,
                    ))
                    return True

        return False

    # =========================================================================
    # Postmortem
    # =========================================================================

    async def create_postmortem(
        self,
        incident_id: str,
        title: str,
        author: str,
    ) -> Optional[Postmortem]:
        """
        Create a postmortem for an incident.

        Args:
            incident_id: Incident ID
            title: Postmortem title
            author: Author

        Returns:
            Created postmortem or None
        """
        incident = await self.get_incident(incident_id)
        if not incident:
            return None

        postmortem_id = f"pm_{secrets.token_hex(8)}"

        postmortem = Postmortem(
            id=postmortem_id,
            incident_id=incident_id,
            title=title,
        )

        async with self._lock:
            self._postmortems[postmortem_id] = postmortem
            incident.postmortem_id = postmortem_id

        logger.info(f"Created postmortem {postmortem_id} for incident {incident_id}")
        return postmortem

    async def get_postmortem(self, postmortem_id: str) -> Optional[Postmortem]:
        """Get a postmortem by ID."""
        async with self._lock:
            return self._postmortems.get(postmortem_id)

    async def update_postmortem(
        self,
        postmortem_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Postmortem]:
        """
        Update a postmortem.

        Args:
            postmortem_id: Postmortem ID
            updates: Fields to update

        Returns:
            Updated postmortem or None
        """
        async with self._lock:
            postmortem = self._postmortems.get(postmortem_id)
            if not postmortem:
                return None

            for key, value in updates.items():
                if hasattr(postmortem, key):
                    setattr(postmortem, key, value)

            if updates.get("status") == "published" and not postmortem.published_at:
                postmortem.published_at = datetime.utcnow()

        return postmortem

    # =========================================================================
    # Statistics and Reporting
    # =========================================================================

    async def get_statistics(
        self,
        organization_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get incident statistics.

        Args:
            organization_id: Filter by organization
            start_date: Start of period
            end_date: End of period

        Returns:
            Statistics dictionary
        """
        incidents = await self.list_incidents(
            organization_id=organization_id,
            limit=10000,
        )

        # Filter by date range
        if start_date:
            incidents = [i for i in incidents if i.created_at >= start_date]
        if end_date:
            incidents = [i for i in incidents if i.created_at <= end_date]

        # Calculate statistics
        total = len(incidents)
        by_severity = {}
        by_status = {}
        total_duration = 0
        resolved_count = 0
        sla_breached_count = 0
        mttr_values = []

        for incident in incidents:
            # By severity
            sev = incident.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

            # By status
            status = incident.status.value
            by_status[status] = by_status.get(status, 0) + 1

            # Duration stats
            if incident.time_to_resolve_seconds:
                total_duration += incident.time_to_resolve_seconds
                mttr_values.append(incident.time_to_resolve_seconds)
                resolved_count += 1

            # SLA stats
            if incident.sla_breached:
                sla_breached_count += 1

        # Calculate MTTR (Mean Time To Resolution)
        mttr = total_duration / resolved_count if resolved_count > 0 else 0

        return {
            "total_incidents": total,
            "active_incidents": sum(1 for i in incidents if i.is_active),
            "by_severity": by_severity,
            "by_status": by_status,
            "mean_time_to_resolution_seconds": mttr,
            "sla_breached_count": sla_breached_count,
            "sla_compliance_rate": 1 - (sla_breached_count / total) if total > 0 else 1.0,
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            },
        }

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_created(self, handler: Callable) -> None:
        """Register handler for incident creation."""
        self._on_created.append(handler)

    def on_updated(self, handler: Callable) -> None:
        """Register handler for incident updates."""
        self._on_updated.append(handler)

    def on_resolved(self, handler: Callable) -> None:
        """Register handler for incident resolution."""
        self._on_resolved.append(handler)

    def on_escalation(self, handler: Callable) -> None:
        """Register handler for escalations."""
        self._on_escalation.append(handler)

    # =========================================================================
    # Background Monitoring
    # =========================================================================

    async def start(self) -> None:
        """Start incident monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Incident manager started")

    async def stop(self) -> None:
        """Stop incident monitoring."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Incident manager stopped")

    async def _monitoring_loop(self) -> None:
        """Background monitoring for escalations and SLA tracking."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Check for stale incidents
                async with self._lock:
                    for incident in self._incidents.values():
                        if not incident.is_active:
                            continue

                        # Check for escalation
                        if incident.time_since_last_update_seconds > self.config.escalate_after_no_update_seconds:
                            await self._escalate_incident(incident)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in incident monitoring: {e}")

    async def _escalate_incident(self, incident: Incident) -> None:
        """Escalate an incident that hasn't been updated."""
        logger.warning(
            f"Incident {incident.id} has not been updated for "
            f"{incident.time_since_last_update_seconds}s, escalating"
        )

        # Add escalation update
        async with self._lock:
            incident.updates.append(IncidentUpdate(
                id=f"upd_{secrets.token_hex(6)}",
                incident_id=incident.id,
                update_type=UpdateType.MESSAGE,
                message="Incident automatically escalated due to no recent updates",
                author="system",
            ))

        # Emit escalation event
        for handler in self._on_escalation:
            try:
                await handler(incident)
            except Exception as e:
                logger.error(f"Error in escalation handler: {e}")


def create_incident_manager(
    config: Optional[IncidentConfig] = None,
) -> IncidentManager:
    """
    Create an incident manager with default configuration.

    Args:
        config: Optional configuration

    Returns:
        Configured IncidentManager
    """
    return IncidentManager(config)

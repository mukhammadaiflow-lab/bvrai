"""Campaign data models."""

from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


class CampaignStatus(str, Enum):
    """Campaign status."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


class ContactStatus(str, Enum):
    """Contact call status."""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BUSY = "busy"
    NO_ANSWER = "no_answer"
    VOICEMAIL = "voicemail"
    SKIPPED = "skipped"
    RETRY = "retry"


class CallOutcome(str, Enum):
    """Call outcome classification."""
    SUCCESS = "success"
    CALLBACK_REQUESTED = "callback_requested"
    NOT_INTERESTED = "not_interested"
    WRONG_NUMBER = "wrong_number"
    DO_NOT_CALL = "do_not_call"
    VOICEMAIL_LEFT = "voicemail_left"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    FAILED = "failed"
    OTHER = "other"


@dataclass
class CampaignSchedule:
    """Campaign schedule configuration."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    timezone: str = "UTC"
    daily_start_time: Optional[time] = None  # e.g., 9:00 AM
    daily_end_time: Optional[time] = None    # e.g., 5:00 PM
    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    respect_holidays: bool = True
    holiday_calendar: str = "US"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "timezone": self.timezone,
            "daily_start_time": self.daily_start_time.isoformat() if self.daily_start_time else None,
            "daily_end_time": self.daily_end_time.isoformat() if self.daily_end_time else None,
            "days_of_week": self.days_of_week,
            "respect_holidays": self.respect_holidays,
            "holiday_calendar": self.holiday_calendar,
        }


@dataclass
class CampaignSettings:
    """Campaign execution settings."""
    max_concurrent_calls: int = 10
    calls_per_minute: int = 30
    max_attempts_per_contact: int = 3
    retry_delay_minutes: int = 60
    voicemail_detection: bool = True
    leave_voicemail: bool = False
    voicemail_message: Optional[str] = None
    answering_machine_detection: bool = True
    amd_timeout_ms: int = 4000
    max_call_duration_seconds: int = 600
    ring_timeout_seconds: int = 30
    record_calls: bool = False
    do_not_call_check: bool = True
    tcpa_compliant: bool = True
    priority: int = 5  # 1-10, higher = more priority

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignContact:
    """A contact in a campaign."""
    id: str
    campaign_id: str
    phone_number: str
    name: Optional[str] = None
    email: Optional[str] = None
    status: ContactStatus = ContactStatus.PENDING
    call_id: Optional[str] = None
    outcome: Optional[CallOutcome] = None
    attempts: int = 0
    last_attempt_at: Optional[datetime] = None
    next_attempt_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: int = 0
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "status": self.status.value,
            "outcome": self.outcome.value if self.outcome else None,
            "last_attempt_at": self.last_attempt_at.isoformat() if self.last_attempt_at else None,
            "next_attempt_at": self.next_attempt_at.isoformat() if self.next_attempt_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def can_retry(self, max_attempts: int) -> bool:
        """Check if contact can be retried."""
        if self.status not in (ContactStatus.FAILED, ContactStatus.NO_ANSWER, ContactStatus.BUSY, ContactStatus.RETRY):
            return False
        return self.attempts < max_attempts


@dataclass
class CampaignProgress:
    """Campaign progress statistics."""
    total: int = 0
    pending: int = 0
    queued: int = 0
    in_progress: int = 0
    completed: int = 0
    successful: int = 0
    failed: int = 0
    no_answer: int = 0
    busy: int = 0
    voicemail: int = 0
    skipped: int = 0
    retry: int = 0
    percent_complete: float = 0.0
    success_rate: float = 0.0
    avg_duration: float = 0.0
    total_duration: int = 0
    estimated_remaining_minutes: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Campaign:
    """A batch call campaign."""
    id: str
    organization_id: str
    name: str
    description: str = ""
    agent_id: str = ""
    from_number: str = ""
    phone_number_id: Optional[str] = None
    status: CampaignStatus = CampaignStatus.DRAFT
    schedule: CampaignSchedule = field(default_factory=CampaignSchedule)
    settings: CampaignSettings = field(default_factory=CampaignSettings)
    total_contacts: int = 0
    contacts_processed: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_seconds: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "description": self.description,
            "agent_id": self.agent_id,
            "from_number": self.from_number,
            "phone_number_id": self.phone_number_id,
            "status": self.status.value,
            "schedule": self.schedule.to_dict(),
            "settings": self.settings.to_dict(),
            "total_contacts": self.total_contacts,
            "contacts_processed": self.contacts_processed,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_duration_seconds": self.total_duration_seconds,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "canceled_at": self.canceled_at.isoformat() if self.canceled_at else None,
            "created_by": self.created_by,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @property
    def is_active(self) -> bool:
        """Check if campaign is currently active."""
        return self.status in (CampaignStatus.RUNNING, CampaignStatus.SCHEDULED)

    @property
    def can_start(self) -> bool:
        """Check if campaign can be started."""
        return self.status in (CampaignStatus.DRAFT, CampaignStatus.PAUSED)

    @property
    def can_pause(self) -> bool:
        """Check if campaign can be paused."""
        return self.status == CampaignStatus.RUNNING

    @property
    def can_cancel(self) -> bool:
        """Check if campaign can be canceled."""
        return self.status in (CampaignStatus.DRAFT, CampaignStatus.SCHEDULED, CampaignStatus.RUNNING, CampaignStatus.PAUSED)


@dataclass
class CampaignCallResult:
    """Result of a campaign call."""
    contact_id: str
    call_id: str
    status: ContactStatus
    outcome: Optional[CallOutcome]
    duration_seconds: int
    transcript_summary: Optional[str] = None
    sentiment: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class CampaignExportData:
    """Campaign export data."""
    campaign: Campaign
    contacts: List[CampaignContact]
    export_format: str
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ContactImportResult:
    """Result of importing contacts."""
    total_imported: int = 0
    duplicates_skipped: int = 0
    invalid_numbers: int = 0
    errors: List[str] = field(default_factory=list)

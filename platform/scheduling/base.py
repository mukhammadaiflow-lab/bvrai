"""
Scheduling Base Types Module

This module defines core types for appointment scheduling, calendar management,
availability tracking, and booking systems.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)


# =============================================================================
# Enums
# =============================================================================


class AppointmentStatus(str, Enum):
    """Status of an appointment."""

    PENDING = "pending"  # Awaiting confirmation
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"
    COMPLETED = "completed"
    NO_SHOW = "no_show"


class AppointmentType(str, Enum):
    """Types of appointments."""

    CONSULTATION = "consultation"
    FOLLOW_UP = "follow_up"
    DEMO = "demo"
    SUPPORT = "support"
    SALES = "sales"
    INTERVIEW = "interview"
    CALLBACK = "callback"
    CUSTOM = "custom"


class RecurrencePattern(str, Enum):
    """Recurrence patterns for schedules."""

    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class DayOfWeek(str, Enum):
    """Days of the week."""

    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


class TimeSlotStatus(str, Enum):
    """Status of a time slot."""

    AVAILABLE = "available"
    BOOKED = "booked"
    BLOCKED = "blocked"
    TENTATIVE = "tentative"


class ReminderChannel(str, Enum):
    """Channels for appointment reminders."""

    EMAIL = "email"
    SMS = "sms"
    VOICE = "voice"
    PUSH = "push"


class CalendarProvider(str, Enum):
    """Calendar integration providers."""

    INTERNAL = "internal"
    GOOGLE = "google"
    OUTLOOK = "outlook"
    APPLE = "apple"
    CALDAV = "caldav"


# =============================================================================
# Time and Schedule Types
# =============================================================================


@dataclass
class TimeRange:
    """A time range within a day."""

    start_time: time
    end_time: time

    @property
    def duration_minutes(self) -> int:
        """Get duration in minutes."""
        start_minutes = self.start_time.hour * 60 + self.start_time.minute
        end_minutes = self.end_time.hour * 60 + self.end_time.minute
        return end_minutes - start_minutes

    def contains(self, t: time) -> bool:
        """Check if time is within range."""
        return self.start_time <= t < self.end_time

    def overlaps(self, other: "TimeRange") -> bool:
        """Check if ranges overlap."""
        return self.start_time < other.end_time and other.start_time < self.end_time

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "TimeRange":
        """Create from dictionary."""
        return cls(
            start_time=time.fromisoformat(data["start_time"]),
            end_time=time.fromisoformat(data["end_time"]),
        )


@dataclass
class BusinessHours:
    """Business hours configuration."""

    day: DayOfWeek
    is_open: bool = True
    time_ranges: List[TimeRange] = field(default_factory=list)

    def is_time_available(self, t: time) -> bool:
        """Check if time is within business hours."""
        if not self.is_open:
            return False
        for tr in self.time_ranges:
            if tr.contains(t):
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "day": self.day.value,
            "is_open": self.is_open,
            "time_ranges": [tr.to_dict() for tr in self.time_ranges],
        }


@dataclass
class Schedule:
    """A schedule defining availability."""

    id: str
    organization_id: str
    name: str

    # Business hours by day
    business_hours: Dict[DayOfWeek, BusinessHours] = field(default_factory=dict)

    # Timezone
    timezone: str = "UTC"

    # Buffer times
    buffer_before_minutes: int = 0
    buffer_after_minutes: int = 0

    # Slot configuration
    slot_duration_minutes: int = 30
    min_notice_hours: int = 24  # Minimum advance booking time

    # Booking limits
    max_bookings_per_day: Optional[int] = None
    max_bookings_per_slot: int = 1

    # Metadata
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"sched_{uuid.uuid4().hex[:18]}"

        # Initialize default business hours if not set
        if not self.business_hours:
            for day in [DayOfWeek.MONDAY, DayOfWeek.TUESDAY, DayOfWeek.WEDNESDAY,
                        DayOfWeek.THURSDAY, DayOfWeek.FRIDAY]:
                self.business_hours[day] = BusinessHours(
                    day=day,
                    is_open=True,
                    time_ranges=[TimeRange(time(9, 0), time(17, 0))],
                )
            for day in [DayOfWeek.SATURDAY, DayOfWeek.SUNDAY]:
                self.business_hours[day] = BusinessHours(day=day, is_open=False)

    def is_day_open(self, d: date) -> bool:
        """Check if day is open."""
        day_of_week = DayOfWeek(d.strftime("%A").lower())
        bh = self.business_hours.get(day_of_week)
        return bh is not None and bh.is_open

    def is_time_available_on_date(self, dt: datetime) -> bool:
        """Check if datetime is within business hours."""
        day_of_week = DayOfWeek(dt.strftime("%A").lower())
        bh = self.business_hours.get(day_of_week)
        if not bh or not bh.is_open:
            return False
        return bh.is_time_available(dt.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "business_hours": {
                day.value: bh.to_dict()
                for day, bh in self.business_hours.items()
            },
            "timezone": self.timezone,
            "buffer_before_minutes": self.buffer_before_minutes,
            "buffer_after_minutes": self.buffer_after_minutes,
            "slot_duration_minutes": self.slot_duration_minutes,
            "min_notice_hours": self.min_notice_hours,
            "max_bookings_per_day": self.max_bookings_per_day,
            "max_bookings_per_slot": self.max_bookings_per_slot,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# Time Slot Types
# =============================================================================


@dataclass
class TimeSlot:
    """A bookable time slot."""

    id: str
    schedule_id: str
    date: date
    start_time: datetime
    end_time: datetime

    # Status
    status: TimeSlotStatus = TimeSlotStatus.AVAILABLE
    bookings_count: int = 0
    max_bookings: int = 1

    # Associated data
    appointment_id: Optional[str] = None
    resource_id: Optional[str] = None  # e.g., specific agent

    def __post_init__(self):
        if not self.id:
            self.id = f"slot_{uuid.uuid4().hex[:18]}"

    @property
    def is_available(self) -> bool:
        """Check if slot is available for booking."""
        if self.status != TimeSlotStatus.AVAILABLE:
            return False
        return self.bookings_count < self.max_bookings

    @property
    def duration_minutes(self) -> int:
        """Get slot duration in minutes."""
        return int((self.end_time - self.start_time).total_seconds() / 60)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "schedule_id": self.schedule_id,
            "date": self.date.isoformat(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "status": self.status.value,
            "is_available": self.is_available,
            "bookings_count": self.bookings_count,
            "max_bookings": self.max_bookings,
            "appointment_id": self.appointment_id,
            "resource_id": self.resource_id,
        }


@dataclass
class BlockedTime:
    """A blocked time period (holiday, vacation, etc.)."""

    id: str
    schedule_id: str
    start_datetime: datetime
    end_datetime: datetime

    # Recurrence
    recurrence: RecurrencePattern = RecurrencePattern.NONE
    recurrence_until: Optional[date] = None

    # Description
    reason: str = ""
    is_all_day: bool = False

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"block_{uuid.uuid4().hex[:18]}"

    def blocks(self, dt: datetime) -> bool:
        """Check if datetime is blocked."""
        if self.is_all_day:
            return self.start_datetime.date() <= dt.date() <= self.end_datetime.date()
        return self.start_datetime <= dt < self.end_datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "schedule_id": self.schedule_id,
            "start_datetime": self.start_datetime.isoformat(),
            "end_datetime": self.end_datetime.isoformat(),
            "recurrence": self.recurrence.value,
            "recurrence_until": self.recurrence_until.isoformat() if self.recurrence_until else None,
            "reason": self.reason,
            "is_all_day": self.is_all_day,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Appointment Types
# =============================================================================


@dataclass
class Participant:
    """An appointment participant."""

    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    is_organizer: bool = False
    is_required: bool = True
    confirmed: bool = False
    confirmed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "is_organizer": self.is_organizer,
            "is_required": self.is_required,
            "confirmed": self.confirmed,
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
        }


@dataclass
class Reminder:
    """An appointment reminder."""

    id: str
    channel: ReminderChannel
    minutes_before: int  # Minutes before appointment

    # Status
    sent: bool = False
    sent_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"rem_{uuid.uuid4().hex[:16]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "channel": self.channel.value,
            "minutes_before": self.minutes_before,
            "sent": self.sent,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "error_message": self.error_message,
        }


@dataclass
class Appointment:
    """An appointment/booking."""

    id: str
    organization_id: str
    schedule_id: str

    # Type and title
    appointment_type: AppointmentType = AppointmentType.CONSULTATION
    title: str = ""
    description: str = ""

    # Time
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime = field(default_factory=datetime.utcnow)
    timezone: str = "UTC"

    # Status
    status: AppointmentStatus = AppointmentStatus.PENDING
    confirmed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    cancellation_reason: Optional[str] = None

    # Participants
    participants: List[Participant] = field(default_factory=list)

    # Customer info (if not participant)
    customer_id: Optional[str] = None
    customer_name: str = ""
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None

    # Resource assignment
    assigned_agent_id: Optional[str] = None
    assigned_resource_id: Optional[str] = None

    # Reminders
    reminders: List[Reminder] = field(default_factory=list)

    # Location/Meeting info
    location: str = ""
    meeting_url: Optional[str] = None
    meeting_id: Optional[str] = None
    meeting_password: Optional[str] = None

    # Notes and context
    notes: str = ""
    internal_notes: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    # Recurrence
    recurrence: RecurrencePattern = RecurrencePattern.NONE
    recurrence_rule: Optional[str] = None  # iCal RRULE
    recurring_appointment_id: Optional[str] = None  # Parent appointment ID

    # External integrations
    external_calendar_id: Optional[str] = None
    external_event_id: Optional[str] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"appt_{uuid.uuid4().hex[:18]}"

    @property
    def duration_minutes(self) -> int:
        """Get appointment duration in minutes."""
        return int((self.end_time - self.start_time).total_seconds() / 60)

    @property
    def is_confirmed(self) -> bool:
        """Check if appointment is confirmed."""
        return self.status == AppointmentStatus.CONFIRMED

    @property
    def is_cancelled(self) -> bool:
        """Check if appointment is cancelled."""
        return self.status == AppointmentStatus.CANCELLED

    @property
    def is_past(self) -> bool:
        """Check if appointment is in the past."""
        return self.end_time < datetime.utcnow()

    @property
    def is_upcoming(self) -> bool:
        """Check if appointment is upcoming."""
        return self.start_time > datetime.utcnow()

    @property
    def time_until(self) -> Optional[timedelta]:
        """Get time until appointment starts."""
        if self.is_past:
            return None
        return self.start_time - datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "schedule_id": self.schedule_id,
            "appointment_type": self.appointment_type.value,
            "title": self.title,
            "description": self.description,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "timezone": self.timezone,
            "duration_minutes": self.duration_minutes,
            "status": self.status.value,
            "is_confirmed": self.is_confirmed,
            "is_upcoming": self.is_upcoming,
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "cancellation_reason": self.cancellation_reason,
            "participants": [p.to_dict() for p in self.participants],
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "customer_email": self.customer_email,
            "customer_phone": self.customer_phone,
            "assigned_agent_id": self.assigned_agent_id,
            "reminders": [r.to_dict() for r in self.reminders],
            "location": self.location,
            "meeting_url": self.meeting_url,
            "notes": self.notes,
            "recurrence": self.recurrence.value,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# Calendar Integration Types
# =============================================================================


@dataclass
class CalendarConnection:
    """Connection to external calendar."""

    id: str
    organization_id: str
    user_id: str
    provider: CalendarProvider

    # Connection details
    calendar_id: str = ""
    calendar_name: str = ""
    access_token: str = ""
    refresh_token: str = ""
    token_expires_at: Optional[datetime] = None

    # Sync settings
    sync_enabled: bool = True
    sync_two_way: bool = True
    last_synced_at: Optional[datetime] = None

    # Status
    is_connected: bool = True
    connection_error: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"cal_{uuid.uuid4().hex[:18]}"

    @property
    def needs_refresh(self) -> bool:
        """Check if token needs refresh."""
        if not self.token_expires_at:
            return False
        return datetime.utcnow() >= self.token_expires_at - timedelta(minutes=5)

    def to_dict(self, include_tokens: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "provider": self.provider.value,
            "calendar_id": self.calendar_id,
            "calendar_name": self.calendar_name,
            "sync_enabled": self.sync_enabled,
            "sync_two_way": self.sync_two_way,
            "last_synced_at": self.last_synced_at.isoformat() if self.last_synced_at else None,
            "is_connected": self.is_connected,
            "connection_error": self.connection_error,
            "created_at": self.created_at.isoformat(),
        }
        if include_tokens:
            data["access_token"] = self.access_token
            data["refresh_token"] = self.refresh_token
        return data


# =============================================================================
# Booking Types
# =============================================================================


@dataclass
class BookingType:
    """A type of booking that can be scheduled."""

    id: str
    organization_id: str
    name: str
    description: str = ""

    # Duration
    duration_minutes: int = 30
    buffer_before_minutes: int = 0
    buffer_after_minutes: int = 0

    # Availability
    schedule_id: Optional[str] = None
    available_days: List[DayOfWeek] = field(
        default_factory=lambda: [
            DayOfWeek.MONDAY, DayOfWeek.TUESDAY, DayOfWeek.WEDNESDAY,
            DayOfWeek.THURSDAY, DayOfWeek.FRIDAY
        ]
    )

    # Booking rules
    min_notice_hours: int = 24
    max_advance_days: int = 60
    max_per_day: Optional[int] = None

    # Confirmation
    requires_confirmation: bool = False
    auto_confirm: bool = True

    # Reminders
    reminder_minutes: List[int] = field(default_factory=lambda: [1440, 60])  # 24h and 1h

    # Location
    default_location: str = ""
    is_virtual: bool = False
    meeting_provider: Optional[str] = None  # zoom, google_meet, teams

    # Customization
    color: str = "#3B82F6"
    is_public: bool = True
    booking_url: Optional[str] = None

    # Metadata
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"btype_{uuid.uuid4().hex[:16]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "description": self.description,
            "duration_minutes": self.duration_minutes,
            "buffer_before_minutes": self.buffer_before_minutes,
            "buffer_after_minutes": self.buffer_after_minutes,
            "schedule_id": self.schedule_id,
            "available_days": [d.value for d in self.available_days],
            "min_notice_hours": self.min_notice_hours,
            "max_advance_days": self.max_advance_days,
            "requires_confirmation": self.requires_confirmation,
            "auto_confirm": self.auto_confirm,
            "reminder_minutes": self.reminder_minutes,
            "default_location": self.default_location,
            "is_virtual": self.is_virtual,
            "color": self.color,
            "is_public": self.is_public,
            "booking_url": self.booking_url,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Exceptions
# =============================================================================


class SchedulingError(Exception):
    """Base exception for scheduling errors."""
    pass


class SlotNotAvailableError(SchedulingError):
    """Time slot is not available."""
    pass


class AppointmentNotFoundError(SchedulingError):
    """Appointment not found."""
    pass


class ScheduleNotFoundError(SchedulingError):
    """Schedule not found."""
    pass


class CalendarSyncError(SchedulingError):
    """Error syncing with external calendar."""
    pass


class BookingConflictError(SchedulingError):
    """Booking conflicts with existing appointment."""
    pass


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "AppointmentStatus",
    "AppointmentType",
    "RecurrencePattern",
    "DayOfWeek",
    "TimeSlotStatus",
    "ReminderChannel",
    "CalendarProvider",
    # Time types
    "TimeRange",
    "BusinessHours",
    "Schedule",
    # Slot types
    "TimeSlot",
    "BlockedTime",
    # Appointment types
    "Participant",
    "Reminder",
    "Appointment",
    # Calendar types
    "CalendarConnection",
    # Booking types
    "BookingType",
    # Exceptions
    "SchedulingError",
    "SlotNotAvailableError",
    "AppointmentNotFoundError",
    "ScheduleNotFoundError",
    "CalendarSyncError",
    "BookingConflictError",
]

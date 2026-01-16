"""
Scheduling Service Module

This module provides comprehensive appointment scheduling, calendar management,
and booking services for voice agent platforms.
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
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

from .base import (
    # Enums
    AppointmentStatus,
    AppointmentType,
    CalendarProvider,
    DayOfWeek,
    RecurrencePattern,
    ReminderChannel,
    TimeSlotStatus,
    # Types
    Appointment,
    BlockedTime,
    BookingType,
    BusinessHours,
    CalendarConnection,
    Participant,
    Reminder,
    Schedule,
    TimeRange,
    TimeSlot,
    # Exceptions
    AppointmentNotFoundError,
    BookingConflictError,
    CalendarSyncError,
    ScheduleNotFoundError,
    SchedulingError,
    SlotNotAvailableError,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Schedule Manager
# =============================================================================


class ScheduleManager:
    """Manages schedules and business hours."""

    def __init__(self):
        self._schedules: Dict[str, Schedule] = {}
        self._schedules_by_org: Dict[str, Set[str]] = defaultdict(set)
        self._blocked_times: Dict[str, List[BlockedTime]] = defaultdict(list)

    async def create_schedule(
        self,
        organization_id: str,
        name: str,
        timezone: str = "UTC",
        slot_duration_minutes: int = 30,
        min_notice_hours: int = 24,
        business_hours: Optional[Dict[DayOfWeek, BusinessHours]] = None,
    ) -> Schedule:
        """Create a new schedule."""
        schedule = Schedule(
            id=f"sched_{uuid.uuid4().hex[:18]}",
            organization_id=organization_id,
            name=name,
            timezone=timezone,
            slot_duration_minutes=slot_duration_minutes,
            min_notice_hours=min_notice_hours,
        )

        if business_hours:
            schedule.business_hours = business_hours

        self._schedules[schedule.id] = schedule
        self._schedules_by_org[organization_id].add(schedule.id)

        logger.info(f"Created schedule {schedule.id}: {name}")

        return schedule

    async def get_schedule(self, schedule_id: str) -> Optional[Schedule]:
        """Get schedule by ID."""
        return self._schedules.get(schedule_id)

    async def list_schedules(
        self,
        organization_id: str,
        active_only: bool = True,
    ) -> List[Schedule]:
        """List schedules for organization."""
        schedule_ids = self._schedules_by_org.get(organization_id, set())
        schedules = []

        for schedule_id in schedule_ids:
            schedule = self._schedules.get(schedule_id)
            if schedule:
                if active_only and not schedule.is_active:
                    continue
                schedules.append(schedule)

        return sorted(schedules, key=lambda s: s.name)

    async def update_schedule(
        self,
        schedule_id: str,
        name: Optional[str] = None,
        slot_duration_minutes: Optional[int] = None,
        min_notice_hours: Optional[int] = None,
        is_active: Optional[bool] = None,
    ) -> Schedule:
        """Update schedule settings."""
        schedule = await self.get_schedule(schedule_id)
        if not schedule:
            raise ScheduleNotFoundError(f"Schedule {schedule_id} not found")

        if name:
            schedule.name = name
        if slot_duration_minutes:
            schedule.slot_duration_minutes = slot_duration_minutes
        if min_notice_hours is not None:
            schedule.min_notice_hours = min_notice_hours
        if is_active is not None:
            schedule.is_active = is_active

        schedule.updated_at = datetime.utcnow()

        return schedule

    async def set_business_hours(
        self,
        schedule_id: str,
        day: DayOfWeek,
        is_open: bool,
        time_ranges: Optional[List[TimeRange]] = None,
    ) -> Schedule:
        """Set business hours for a day."""
        schedule = await self.get_schedule(schedule_id)
        if not schedule:
            raise ScheduleNotFoundError(f"Schedule {schedule_id} not found")

        schedule.business_hours[day] = BusinessHours(
            day=day,
            is_open=is_open,
            time_ranges=time_ranges or [],
        )
        schedule.updated_at = datetime.utcnow()

        return schedule

    async def add_blocked_time(
        self,
        schedule_id: str,
        start_datetime: datetime,
        end_datetime: datetime,
        reason: str = "",
        is_all_day: bool = False,
        recurrence: RecurrencePattern = RecurrencePattern.NONE,
    ) -> BlockedTime:
        """Add a blocked time period."""
        blocked = BlockedTime(
            id=f"block_{uuid.uuid4().hex[:18]}",
            schedule_id=schedule_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            reason=reason,
            is_all_day=is_all_day,
            recurrence=recurrence,
        )

        self._blocked_times[schedule_id].append(blocked)

        logger.info(f"Added blocked time {blocked.id} for schedule {schedule_id}")

        return blocked

    async def get_blocked_times(
        self,
        schedule_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[BlockedTime]:
        """Get blocked times for schedule."""
        blocked_list = self._blocked_times.get(schedule_id, [])

        if start_date or end_date:
            filtered = []
            for blocked in blocked_list:
                if start_date and blocked.end_datetime.date() < start_date:
                    continue
                if end_date and blocked.start_datetime.date() > end_date:
                    continue
                filtered.append(blocked)
            return filtered

        return blocked_list

    async def remove_blocked_time(self, blocked_id: str) -> bool:
        """Remove a blocked time."""
        for schedule_id, blocked_list in self._blocked_times.items():
            for i, blocked in enumerate(blocked_list):
                if blocked.id == blocked_id:
                    del self._blocked_times[schedule_id][i]
                    return True
        return False

    async def is_time_available(
        self,
        schedule_id: str,
        dt: datetime,
    ) -> bool:
        """Check if datetime is available."""
        schedule = await self.get_schedule(schedule_id)
        if not schedule or not schedule.is_active:
            return False

        # Check business hours
        if not schedule.is_time_available_on_date(dt):
            return False

        # Check blocked times
        blocked_times = await self.get_blocked_times(schedule_id)
        for blocked in blocked_times:
            if blocked.blocks(dt):
                return False

        return True


# =============================================================================
# Slot Generator
# =============================================================================


class SlotGenerator:
    """Generates available time slots."""

    def __init__(
        self,
        schedule_manager: ScheduleManager,
    ):
        self._schedule_manager = schedule_manager
        self._booked_slots: Dict[str, Set[str]] = defaultdict(set)  # schedule_id -> slot keys
        self._booking_lock = asyncio.Lock()  # Lock for booking operations

    def _slot_key(self, dt: datetime, duration: int) -> str:
        """Generate unique key for slot."""
        return f"{dt.isoformat()}:{duration}"

    async def generate_slots(
        self,
        schedule_id: str,
        start_date: date,
        end_date: date,
        duration_minutes: Optional[int] = None,
    ) -> List[TimeSlot]:
        """Generate available time slots for date range."""
        schedule = await self._schedule_manager.get_schedule(schedule_id)
        if not schedule:
            return []

        duration = duration_minutes or schedule.slot_duration_minutes
        slots = []

        current_date = start_date
        while current_date <= end_date:
            day_slots = await self._generate_day_slots(
                schedule, current_date, duration
            )
            slots.extend(day_slots)
            current_date += timedelta(days=1)

        return slots

    async def _generate_day_slots(
        self,
        schedule: Schedule,
        d: date,
        duration_minutes: int,
    ) -> List[TimeSlot]:
        """Generate slots for a single day."""
        day_of_week = DayOfWeek(d.strftime("%A").lower())
        business_hours = schedule.business_hours.get(day_of_week)

        if not business_hours or not business_hours.is_open:
            return []

        slots = []
        for time_range in business_hours.time_ranges:
            range_slots = await self._generate_range_slots(
                schedule, d, time_range, duration_minutes
            )
            slots.extend(range_slots)

        return slots

    async def _generate_range_slots(
        self,
        schedule: Schedule,
        d: date,
        time_range: TimeRange,
        duration_minutes: int,
    ) -> List[TimeSlot]:
        """Generate slots for a time range."""
        slots = []
        current = datetime.combine(d, time_range.start_time)
        end = datetime.combine(d, time_range.end_time)

        while current + timedelta(minutes=duration_minutes) <= end:
            slot_end = current + timedelta(minutes=duration_minutes)

            # Check if slot is available
            is_available = await self._schedule_manager.is_time_available(
                schedule.id, current
            )

            # Check minimum notice
            min_notice = datetime.utcnow() + timedelta(hours=schedule.min_notice_hours)
            if current < min_notice:
                is_available = False

            # Check if already booked
            slot_key = self._slot_key(current, duration_minutes)
            if slot_key in self._booked_slots[schedule.id]:
                is_available = False

            slot = TimeSlot(
                id=f"slot_{uuid.uuid4().hex[:18]}",
                schedule_id=schedule.id,
                date=d,
                start_time=current,
                end_time=slot_end,
                status=TimeSlotStatus.AVAILABLE if is_available else TimeSlotStatus.BOOKED,
            )
            slots.append(slot)

            current = slot_end + timedelta(minutes=schedule.buffer_after_minutes)

        return slots

    async def book_slot(
        self,
        schedule_id: str,
        start_time: datetime,
        duration_minutes: int,
    ) -> bool:
        """Mark a slot as booked (thread-safe)."""
        slot_key = self._slot_key(start_time, duration_minutes)
        async with self._booking_lock:
            if slot_key in self._booked_slots[schedule_id]:
                return False
            self._booked_slots[schedule_id].add(slot_key)
            return True

    async def release_slot(
        self,
        schedule_id: str,
        start_time: datetime,
        duration_minutes: int,
    ) -> bool:
        """Release a booked slot (thread-safe)."""
        slot_key = self._slot_key(start_time, duration_minutes)
        async with self._booking_lock:
            if slot_key in self._booked_slots[schedule_id]:
                self._booked_slots[schedule_id].remove(slot_key)
                return True
            return False

    async def get_available_slots(
        self,
        schedule_id: str,
        start_date: date,
        end_date: date,
        duration_minutes: Optional[int] = None,
    ) -> List[TimeSlot]:
        """Get only available slots."""
        all_slots = await self.generate_slots(
            schedule_id, start_date, end_date, duration_minutes
        )
        return [s for s in all_slots if s.is_available]


# =============================================================================
# Appointment Manager
# =============================================================================


class AppointmentManager:
    """Manages appointments and bookings."""

    def __init__(
        self,
        schedule_manager: ScheduleManager,
        slot_generator: SlotGenerator,
    ):
        self._schedule_manager = schedule_manager
        self._slot_generator = slot_generator
        self._appointments: Dict[str, Appointment] = {}
        self._appointments_by_org: Dict[str, Set[str]] = defaultdict(set)
        self._appointments_by_schedule: Dict[str, Set[str]] = defaultdict(set)
        self._appointments_by_customer: Dict[str, Set[str]] = defaultdict(set)

    async def create_appointment(
        self,
        organization_id: str,
        schedule_id: str,
        start_time: datetime,
        end_time: datetime,
        appointment_type: AppointmentType = AppointmentType.CONSULTATION,
        title: str = "",
        customer_name: str = "",
        customer_email: Optional[str] = None,
        customer_phone: Optional[str] = None,
        customer_id: Optional[str] = None,
        assigned_agent_id: Optional[str] = None,
        notes: str = "",
        auto_confirm: bool = True,
    ) -> Appointment:
        """Create a new appointment."""
        # Verify slot availability
        is_available = await self._schedule_manager.is_time_available(
            schedule_id, start_time
        )
        if not is_available:
            raise SlotNotAvailableError(
                f"Time slot at {start_time} is not available"
            )

        # Check for conflicts
        conflicts = await self._check_conflicts(
            schedule_id, start_time, end_time
        )
        if conflicts:
            raise BookingConflictError(
                f"Time slot conflicts with {len(conflicts)} existing appointments"
            )

        # Create appointment
        appointment = Appointment(
            id=f"appt_{uuid.uuid4().hex[:18]}",
            organization_id=organization_id,
            schedule_id=schedule_id,
            appointment_type=appointment_type,
            title=title or f"{appointment_type.value.title()} Appointment",
            start_time=start_time,
            end_time=end_time,
            customer_name=customer_name,
            customer_email=customer_email,
            customer_phone=customer_phone,
            customer_id=customer_id,
            assigned_agent_id=assigned_agent_id,
            notes=notes,
            status=AppointmentStatus.CONFIRMED if auto_confirm else AppointmentStatus.PENDING,
            confirmed_at=datetime.utcnow() if auto_confirm else None,
        )

        # Set up default reminders
        appointment.reminders = [
            Reminder(
                id=f"rem_{uuid.uuid4().hex[:16]}",
                channel=ReminderChannel.EMAIL,
                minutes_before=1440,  # 24 hours
            ),
            Reminder(
                id=f"rem_{uuid.uuid4().hex[:16]}",
                channel=ReminderChannel.SMS,
                minutes_before=60,  # 1 hour
            ),
        ]

        # Store
        self._appointments[appointment.id] = appointment
        self._appointments_by_org[organization_id].add(appointment.id)
        self._appointments_by_schedule[schedule_id].add(appointment.id)
        if customer_id:
            self._appointments_by_customer[customer_id].add(appointment.id)

        # Book slot
        duration = int((end_time - start_time).total_seconds() / 60)
        await self._slot_generator.book_slot(schedule_id, start_time, duration)

        logger.info(f"Created appointment {appointment.id}")

        return appointment

    async def _check_conflicts(
        self,
        schedule_id: str,
        start_time: datetime,
        end_time: datetime,
        exclude_id: Optional[str] = None,
    ) -> List[Appointment]:
        """Check for conflicting appointments."""
        conflicts = []
        appointment_ids = self._appointments_by_schedule.get(schedule_id, set())

        for apt_id in appointment_ids:
            if apt_id == exclude_id:
                continue

            apt = self._appointments.get(apt_id)
            if not apt or apt.is_cancelled:
                continue

            # Check overlap
            if start_time < apt.end_time and end_time > apt.start_time:
                conflicts.append(apt)

        return conflicts

    async def get_appointment(
        self,
        appointment_id: str,
    ) -> Optional[Appointment]:
        """Get appointment by ID."""
        return self._appointments.get(appointment_id)

    async def list_appointments(
        self,
        organization_id: str,
        schedule_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        status: Optional[AppointmentStatus] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        upcoming_only: bool = False,
    ) -> List[Appointment]:
        """List appointments with filters."""
        if customer_id:
            apt_ids = self._appointments_by_customer.get(customer_id, set())
        elif schedule_id:
            apt_ids = self._appointments_by_schedule.get(schedule_id, set())
        else:
            apt_ids = self._appointments_by_org.get(organization_id, set())

        appointments = []
        for apt_id in apt_ids:
            apt = self._appointments.get(apt_id)
            if not apt:
                continue

            if status and apt.status != status:
                continue
            if start_date and apt.start_time.date() < start_date:
                continue
            if end_date and apt.start_time.date() > end_date:
                continue
            if upcoming_only and not apt.is_upcoming:
                continue

            appointments.append(apt)

        return sorted(appointments, key=lambda a: a.start_time)

    async def confirm_appointment(
        self,
        appointment_id: str,
    ) -> Appointment:
        """Confirm a pending appointment."""
        appointment = await self.get_appointment(appointment_id)
        if not appointment:
            raise AppointmentNotFoundError(f"Appointment {appointment_id} not found")

        if appointment.status == AppointmentStatus.CANCELLED:
            raise SchedulingError("Cannot confirm cancelled appointment")

        appointment.status = AppointmentStatus.CONFIRMED
        appointment.confirmed_at = datetime.utcnow()
        appointment.updated_at = datetime.utcnow()

        logger.info(f"Confirmed appointment {appointment_id}")

        return appointment

    async def cancel_appointment(
        self,
        appointment_id: str,
        reason: str = "",
    ) -> Appointment:
        """Cancel an appointment."""
        appointment = await self.get_appointment(appointment_id)
        if not appointment:
            raise AppointmentNotFoundError(f"Appointment {appointment_id} not found")

        if appointment.status == AppointmentStatus.CANCELLED:
            return appointment

        appointment.status = AppointmentStatus.CANCELLED
        appointment.cancelled_at = datetime.utcnow()
        appointment.cancellation_reason = reason
        appointment.updated_at = datetime.utcnow()

        # Release slot
        duration = appointment.duration_minutes
        await self._slot_generator.release_slot(
            appointment.schedule_id,
            appointment.start_time,
            duration,
        )

        logger.info(f"Cancelled appointment {appointment_id}: {reason}")

        return appointment

    async def reschedule_appointment(
        self,
        appointment_id: str,
        new_start_time: datetime,
        new_end_time: datetime,
    ) -> Appointment:
        """Reschedule an appointment."""
        appointment = await self.get_appointment(appointment_id)
        if not appointment:
            raise AppointmentNotFoundError(f"Appointment {appointment_id} not found")

        # Check new slot availability
        is_available = await self._schedule_manager.is_time_available(
            appointment.schedule_id, new_start_time
        )
        if not is_available:
            raise SlotNotAvailableError(
                f"Time slot at {new_start_time} is not available"
            )

        # Check for conflicts
        conflicts = await self._check_conflicts(
            appointment.schedule_id,
            new_start_time,
            new_end_time,
            exclude_id=appointment_id,
        )
        if conflicts:
            raise BookingConflictError("Time slot conflicts with existing appointment")

        # Release old slot
        old_duration = appointment.duration_minutes
        await self._slot_generator.release_slot(
            appointment.schedule_id,
            appointment.start_time,
            old_duration,
        )

        # Update appointment
        appointment.start_time = new_start_time
        appointment.end_time = new_end_time
        appointment.status = AppointmentStatus.RESCHEDULED
        appointment.updated_at = datetime.utcnow()

        # Book new slot
        new_duration = int((new_end_time - new_start_time).total_seconds() / 60)
        await self._slot_generator.book_slot(
            appointment.schedule_id,
            new_start_time,
            new_duration,
        )

        logger.info(f"Rescheduled appointment {appointment_id} to {new_start_time}")

        return appointment

    async def complete_appointment(
        self,
        appointment_id: str,
        notes: str = "",
    ) -> Appointment:
        """Mark appointment as completed."""
        appointment = await self.get_appointment(appointment_id)
        if not appointment:
            raise AppointmentNotFoundError(f"Appointment {appointment_id} not found")

        appointment.status = AppointmentStatus.COMPLETED
        if notes:
            appointment.internal_notes = notes
        appointment.updated_at = datetime.utcnow()

        logger.info(f"Completed appointment {appointment_id}")

        return appointment

    async def mark_no_show(
        self,
        appointment_id: str,
    ) -> Appointment:
        """Mark appointment as no-show."""
        appointment = await self.get_appointment(appointment_id)
        if not appointment:
            raise AppointmentNotFoundError(f"Appointment {appointment_id} not found")

        appointment.status = AppointmentStatus.NO_SHOW
        appointment.updated_at = datetime.utcnow()

        logger.info(f"Marked appointment {appointment_id} as no-show")

        return appointment

    async def add_participant(
        self,
        appointment_id: str,
        name: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        is_required: bool = True,
    ) -> Appointment:
        """Add participant to appointment."""
        appointment = await self.get_appointment(appointment_id)
        if not appointment:
            raise AppointmentNotFoundError(f"Appointment {appointment_id} not found")

        participant = Participant(
            name=name,
            email=email,
            phone=phone,
            is_required=is_required,
        )
        appointment.participants.append(participant)
        appointment.updated_at = datetime.utcnow()

        return appointment


# =============================================================================
# Booking Type Manager
# =============================================================================


class BookingTypeManager:
    """Manages booking types."""

    def __init__(self):
        self._booking_types: Dict[str, BookingType] = {}
        self._booking_types_by_org: Dict[str, Set[str]] = defaultdict(set)

    async def create_booking_type(
        self,
        organization_id: str,
        name: str,
        description: str = "",
        duration_minutes: int = 30,
        schedule_id: Optional[str] = None,
        is_virtual: bool = False,
        **kwargs,
    ) -> BookingType:
        """Create a booking type."""
        booking_type = BookingType(
            id=f"btype_{uuid.uuid4().hex[:16]}",
            organization_id=organization_id,
            name=name,
            description=description,
            duration_minutes=duration_minutes,
            schedule_id=schedule_id,
            is_virtual=is_virtual,
            **kwargs,
        )

        self._booking_types[booking_type.id] = booking_type
        self._booking_types_by_org[organization_id].add(booking_type.id)

        logger.info(f"Created booking type {booking_type.id}: {name}")

        return booking_type

    async def get_booking_type(
        self,
        booking_type_id: str,
    ) -> Optional[BookingType]:
        """Get booking type by ID."""
        return self._booking_types.get(booking_type_id)

    async def list_booking_types(
        self,
        organization_id: str,
        active_only: bool = True,
        public_only: bool = False,
    ) -> List[BookingType]:
        """List booking types for organization."""
        type_ids = self._booking_types_by_org.get(organization_id, set())
        types = []

        for type_id in type_ids:
            bt = self._booking_types.get(type_id)
            if not bt:
                continue
            if active_only and not bt.is_active:
                continue
            if public_only and not bt.is_public:
                continue
            types.append(bt)

        return sorted(types, key=lambda t: t.name)

    async def update_booking_type(
        self,
        booking_type_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        duration_minutes: Optional[int] = None,
        is_active: Optional[bool] = None,
        is_public: Optional[bool] = None,
    ) -> BookingType:
        """Update a booking type."""
        bt = await self.get_booking_type(booking_type_id)
        if not bt:
            raise SchedulingError(f"Booking type {booking_type_id} not found")

        if name:
            bt.name = name
        if description is not None:
            bt.description = description
        if duration_minutes:
            bt.duration_minutes = duration_minutes
        if is_active is not None:
            bt.is_active = is_active
        if is_public is not None:
            bt.is_public = is_public

        return bt


# =============================================================================
# Reminder Service
# =============================================================================


class ReminderService:
    """Manages appointment reminders."""

    def __init__(self):
        self._pending_reminders: Dict[str, List[Reminder]] = {}
        self._reminder_callbacks: List[Callable[[Appointment, Reminder], None]] = []

    def add_callback(
        self,
        callback: Callable[[Appointment, Reminder], None],
    ) -> None:
        """Add callback for sending reminders."""
        self._reminder_callbacks.append(callback)

    async def schedule_reminders(
        self,
        appointment: Appointment,
    ) -> None:
        """Schedule reminders for appointment."""
        self._pending_reminders[appointment.id] = appointment.reminders.copy()

        logger.info(
            f"Scheduled {len(appointment.reminders)} reminders for "
            f"appointment {appointment.id}"
        )

    async def check_and_send_reminders(
        self,
        appointments: List[Appointment],
    ) -> List[Tuple[Appointment, Reminder]]:
        """Check and send due reminders."""
        sent = []
        now = datetime.utcnow()

        for appointment in appointments:
            if appointment.is_cancelled or appointment.is_past:
                continue

            for reminder in appointment.reminders:
                if reminder.sent:
                    continue

                # Check if reminder is due
                send_at = appointment.start_time - timedelta(
                    minutes=reminder.minutes_before
                )
                if now >= send_at:
                    # Send reminder
                    await self._send_reminder(appointment, reminder)
                    sent.append((appointment, reminder))

        return sent

    async def _send_reminder(
        self,
        appointment: Appointment,
        reminder: Reminder,
    ) -> None:
        """Send a reminder."""
        for callback in self._reminder_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(appointment, reminder)
                else:
                    callback(appointment, reminder)
            except Exception as e:
                reminder.error_message = str(e)
                logger.error(f"Error sending reminder: {e}")
                return

        reminder.sent = True
        reminder.sent_at = datetime.utcnow()

        logger.info(
            f"Sent {reminder.channel.value} reminder for appointment "
            f"{appointment.id}"
        )

    async def cancel_reminders(self, appointment_id: str) -> None:
        """Cancel all pending reminders for appointment."""
        if appointment_id in self._pending_reminders:
            del self._pending_reminders[appointment_id]


# =============================================================================
# Calendar Sync Service
# =============================================================================


class CalendarSyncService:
    """Syncs with external calendars."""

    def __init__(self):
        self._connections: Dict[str, CalendarConnection] = {}
        self._connections_by_user: Dict[str, Set[str]] = defaultdict(set)

    async def connect_calendar(
        self,
        organization_id: str,
        user_id: str,
        provider: CalendarProvider,
        access_token: str,
        refresh_token: str,
        calendar_id: str = "",
        calendar_name: str = "",
    ) -> CalendarConnection:
        """Connect an external calendar."""
        connection = CalendarConnection(
            id=f"cal_{uuid.uuid4().hex[:18]}",
            organization_id=organization_id,
            user_id=user_id,
            provider=provider,
            calendar_id=calendar_id,
            calendar_name=calendar_name,
            access_token=access_token,
            refresh_token=refresh_token,
        )

        self._connections[connection.id] = connection
        self._connections_by_user[user_id].add(connection.id)

        logger.info(f"Connected calendar {connection.id} for user {user_id}")

        return connection

    async def get_connection(
        self,
        connection_id: str,
    ) -> Optional[CalendarConnection]:
        """Get calendar connection by ID."""
        return self._connections.get(connection_id)

    async def list_connections(
        self,
        user_id: str,
    ) -> List[CalendarConnection]:
        """List calendar connections for user."""
        conn_ids = self._connections_by_user.get(user_id, set())
        connections = []

        for conn_id in conn_ids:
            conn = self._connections.get(conn_id)
            if conn:
                connections.append(conn)

        return connections

    async def disconnect_calendar(self, connection_id: str) -> bool:
        """Disconnect a calendar."""
        conn = self._connections.get(connection_id)
        if not conn:
            return False

        del self._connections[connection_id]
        self._connections_by_user[conn.user_id].discard(connection_id)

        logger.info(f"Disconnected calendar {connection_id}")

        return True

    async def sync_appointments(
        self,
        connection_id: str,
        appointments: List[Appointment],
    ) -> int:
        """Sync appointments to external calendar."""
        connection = await self.get_connection(connection_id)
        if not connection or not connection.is_connected:
            raise CalendarSyncError("Calendar not connected")

        synced_count = 0

        for appointment in appointments:
            try:
                await self._sync_appointment_to_provider(connection, appointment)
                synced_count += 1
            except Exception as e:
                logger.error(f"Error syncing appointment {appointment.id}: {e}")

        connection.last_synced_at = datetime.utcnow()

        logger.info(f"Synced {synced_count} appointments to calendar {connection_id}")

        return synced_count

    async def _sync_appointment_to_provider(
        self,
        connection: CalendarConnection,
        appointment: Appointment,
    ) -> None:
        """Sync single appointment to provider."""
        # In real implementation, call provider API
        if connection.provider == CalendarProvider.GOOGLE:
            await self._sync_to_google(connection, appointment)
        elif connection.provider == CalendarProvider.OUTLOOK:
            await self._sync_to_outlook(connection, appointment)
        # etc.

    async def _sync_to_google(
        self,
        connection: CalendarConnection,
        appointment: Appointment,
    ) -> None:
        """Sync to Google Calendar."""
        # In real implementation, use Google Calendar API
        pass

    async def _sync_to_outlook(
        self,
        connection: CalendarConnection,
        appointment: Appointment,
    ) -> None:
        """Sync to Outlook Calendar."""
        # In real implementation, use Microsoft Graph API
        pass


# =============================================================================
# Scheduling Service
# =============================================================================


class SchedulingService:
    """
    Unified scheduling service.

    Provides:
    - Schedule management
    - Appointment booking
    - Slot generation
    - Reminders
    - Calendar sync
    """

    def __init__(self):
        self.schedules = ScheduleManager()
        self.slots = SlotGenerator(self.schedules)
        self.appointments = AppointmentManager(self.schedules, self.slots)
        self.booking_types = BookingTypeManager()
        self.reminders = ReminderService()
        self.calendar_sync = CalendarSyncService()

    async def book_appointment(
        self,
        organization_id: str,
        schedule_id: str,
        start_time: datetime,
        duration_minutes: int,
        customer_name: str,
        customer_email: Optional[str] = None,
        customer_phone: Optional[str] = None,
        appointment_type: AppointmentType = AppointmentType.CONSULTATION,
        title: str = "",
        notes: str = "",
        send_confirmation: bool = True,
    ) -> Appointment:
        """
        Book an appointment.

        This is the main booking entry point.
        """
        end_time = start_time + timedelta(minutes=duration_minutes)

        appointment = await self.appointments.create_appointment(
            organization_id=organization_id,
            schedule_id=schedule_id,
            start_time=start_time,
            end_time=end_time,
            appointment_type=appointment_type,
            title=title,
            customer_name=customer_name,
            customer_email=customer_email,
            customer_phone=customer_phone,
            notes=notes,
        )

        # Schedule reminders
        await self.reminders.schedule_reminders(appointment)

        logger.info(
            f"Booked appointment {appointment.id} for {customer_name} "
            f"at {start_time}"
        )

        return appointment

    async def get_available_slots(
        self,
        schedule_id: str,
        start_date: date,
        end_date: date,
        duration_minutes: Optional[int] = None,
    ) -> List[TimeSlot]:
        """Get available time slots."""
        return await self.slots.get_available_slots(
            schedule_id, start_date, end_date, duration_minutes
        )

    async def cancel_appointment(
        self,
        appointment_id: str,
        reason: str = "",
        notify_customer: bool = True,
    ) -> Appointment:
        """Cancel an appointment."""
        appointment = await self.appointments.cancel_appointment(
            appointment_id, reason
        )

        # Cancel reminders
        await self.reminders.cancel_reminders(appointment_id)

        return appointment

    async def reschedule_appointment(
        self,
        appointment_id: str,
        new_start_time: datetime,
        new_duration_minutes: Optional[int] = None,
    ) -> Appointment:
        """Reschedule an appointment."""
        appointment = await self.appointments.get_appointment(appointment_id)
        if not appointment:
            raise AppointmentNotFoundError(f"Appointment {appointment_id} not found")

        duration = new_duration_minutes or appointment.duration_minutes
        new_end_time = new_start_time + timedelta(minutes=duration)

        updated = await self.appointments.reschedule_appointment(
            appointment_id, new_start_time, new_end_time
        )

        # Reschedule reminders
        await self.reminders.cancel_reminders(appointment_id)
        await self.reminders.schedule_reminders(updated)

        return updated

    async def get_daily_schedule(
        self,
        organization_id: str,
        schedule_id: str,
        d: date,
    ) -> Dict[str, Any]:
        """Get full daily schedule with appointments and slots."""
        schedule = await self.schedules.get_schedule(schedule_id)
        if not schedule:
            raise ScheduleNotFoundError(f"Schedule {schedule_id} not found")

        slots = await self.slots.generate_slots(schedule_id, d, d)
        appointments = await self.appointments.list_appointments(
            organization_id,
            schedule_id=schedule_id,
            start_date=d,
            end_date=d,
        )

        return {
            "date": d.isoformat(),
            "schedule": schedule.to_dict(),
            "slots": [s.to_dict() for s in slots],
            "appointments": [a.to_dict() for a in appointments],
            "available_count": sum(1 for s in slots if s.is_available),
            "booked_count": len(appointments),
        }

    async def get_booking_statistics(
        self,
        organization_id: str,
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        """Get booking statistics."""
        appointments = await self.appointments.list_appointments(
            organization_id,
            start_date=start_date,
            end_date=end_date,
        )

        confirmed = [a for a in appointments if a.status == AppointmentStatus.CONFIRMED]
        cancelled = [a for a in appointments if a.status == AppointmentStatus.CANCELLED]
        completed = [a for a in appointments if a.status == AppointmentStatus.COMPLETED]
        no_shows = [a for a in appointments if a.status == AppointmentStatus.NO_SHOW]

        return {
            "total": len(appointments),
            "confirmed": len(confirmed),
            "cancelled": len(cancelled),
            "completed": len(completed),
            "no_shows": len(no_shows),
            "cancellation_rate": len(cancelled) / len(appointments) if appointments else 0,
            "no_show_rate": len(no_shows) / len(appointments) if appointments else 0,
            "completion_rate": len(completed) / len(appointments) if appointments else 0,
        }


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Managers
    "ScheduleManager",
    "SlotGenerator",
    "AppointmentManager",
    "BookingTypeManager",
    # Services
    "ReminderService",
    "CalendarSyncService",
    # Main Service
    "SchedulingService",
]

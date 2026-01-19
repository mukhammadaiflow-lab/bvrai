"""
Scheduling Module

This module provides comprehensive appointment scheduling, calendar management,
and booking services for voice agent platforms.

Features:
- Schedule Management: Business hours, timezone support, blocked times
- Appointment Booking: Create, confirm, cancel, reschedule appointments
- Time Slot Generation: Automatic slot generation based on availability
- Booking Types: Configurable booking types with different durations
- Reminder System: Multi-channel appointment reminders
- Calendar Integration: Google Calendar, Outlook, Apple Calendar sync

Example usage:

    from bvrai_core.scheduling import (
        SchedulingService,
        AppointmentType,
        DayOfWeek,
        TimeRange,
        BusinessHours,
        ReminderChannel,
    )
    from datetime import datetime, date, time, timedelta

    # Initialize service
    service = SchedulingService()

    # Create a schedule with business hours
    schedule = await service.schedules.create_schedule(
        organization_id="org_123",
        name="Main Schedule",
        timezone="America/New_York",
        slot_duration_minutes=30,
        min_notice_hours=24,
    )

    # Set custom business hours
    await service.schedules.set_business_hours(
        schedule_id=schedule.id,
        day=DayOfWeek.MONDAY,
        is_open=True,
        time_ranges=[
            TimeRange(time(9, 0), time(12, 0)),
            TimeRange(time(13, 0), time(17, 0)),
        ],
    )

    # Block time for holiday
    await service.schedules.add_blocked_time(
        schedule_id=schedule.id,
        start_datetime=datetime(2024, 12, 25),
        end_datetime=datetime(2024, 12, 26),
        reason="Christmas Holiday",
        is_all_day=True,
    )

    # Create a booking type
    booking_type = await service.booking_types.create_booking_type(
        organization_id="org_123",
        name="30-Minute Consultation",
        description="Free consultation call",
        duration_minutes=30,
        schedule_id=schedule.id,
        is_virtual=True,
    )

    # Get available slots
    available_slots = await service.get_available_slots(
        schedule_id=schedule.id,
        start_date=date.today(),
        end_date=date.today() + timedelta(days=7),
        duration_minutes=30,
    )

    # Book an appointment
    appointment = await service.book_appointment(
        organization_id="org_123",
        schedule_id=schedule.id,
        start_time=available_slots[0].start_time,
        duration_minutes=30,
        customer_name="John Doe",
        customer_email="john@example.com",
        customer_phone="+15551234567",
        appointment_type=AppointmentType.CONSULTATION,
        title="Initial Consultation",
    )

    # Confirm appointment
    await service.appointments.confirm_appointment(appointment.id)

    # Reschedule appointment
    new_time = appointment.start_time + timedelta(days=1)
    await service.reschedule_appointment(
        appointment_id=appointment.id,
        new_start_time=new_time,
    )

    # Cancel appointment
    await service.cancel_appointment(
        appointment_id=appointment.id,
        reason="Customer request",
    )

    # Get daily schedule
    daily = await service.get_daily_schedule(
        organization_id="org_123",
        schedule_id=schedule.id,
        d=date.today(),
    )
    print(f"Available slots: {daily['available_count']}")
    print(f"Booked appointments: {daily['booked_count']}")

    # Connect external calendar
    connection = await service.calendar_sync.connect_calendar(
        organization_id="org_123",
        user_id="user_456",
        provider=CalendarProvider.GOOGLE,
        access_token="ya29...",
        refresh_token="1//...",
        calendar_id="primary",
    )

    # Get booking statistics
    stats = await service.get_booking_statistics(
        organization_id="org_123",
        start_date=date.today() - timedelta(days=30),
        end_date=date.today(),
    )
    print(f"Total bookings: {stats['total']}")
    print(f"Completion rate: {stats['completion_rate']:.1%}")
"""

# Base types and enums
from .base import (
    # Enums
    AppointmentStatus,
    AppointmentType,
    RecurrencePattern,
    DayOfWeek,
    TimeSlotStatus,
    ReminderChannel,
    CalendarProvider,
    # Time types
    TimeRange,
    BusinessHours,
    Schedule,
    # Slot types
    TimeSlot,
    BlockedTime,
    # Appointment types
    Participant,
    Reminder,
    Appointment,
    # Calendar types
    CalendarConnection,
    # Booking types
    BookingType,
    # Exceptions
    SchedulingError,
    SlotNotAvailableError,
    AppointmentNotFoundError,
    ScheduleNotFoundError,
    CalendarSyncError,
    BookingConflictError,
)

# Services
from .service import (
    # Managers
    ScheduleManager,
    SlotGenerator,
    AppointmentManager,
    BookingTypeManager,
    # Services
    ReminderService,
    CalendarSyncService,
    # Main Service
    SchedulingService,
)


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
    # Exceptions
    "SchedulingError",
    "SlotNotAvailableError",
    "AppointmentNotFoundError",
    "ScheduleNotFoundError",
    "CalendarSyncError",
    "BookingConflictError",
]

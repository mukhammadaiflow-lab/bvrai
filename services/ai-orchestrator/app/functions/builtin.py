"""Built-in functions for voice AI agents."""

import asyncio
import httpx
import structlog
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from app.functions.registry import FunctionRegistry, default_registry
from app.functions.schema import FunctionParameter, ParameterType


logger = structlog.get_logger()


class BuiltinFunctions:
    """
    Built-in functions available to all agents.

    Includes:
    - Calendar/scheduling functions
    - Information lookup
    - Call control
    - Data collection
    """

    def __init__(self, registry: Optional[FunctionRegistry] = None):
        self.registry = registry or default_registry
        self._http_client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize and register all built-in functions."""
        self._http_client = httpx.AsyncClient(timeout=30.0)
        self._register_all()
        logger.info("builtin_functions_registered")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()

    def _register_all(self) -> None:
        """Register all built-in functions."""

        # =====================================================================
        # Calendar / Scheduling Functions
        # =====================================================================

        self.registry.register(
            name="check_availability",
            description="Check calendar availability for scheduling appointments. Returns available time slots.",
            handler=self.check_availability,
            parameters=[
                FunctionParameter(
                    name="date",
                    type=ParameterType.STRING,
                    description="Date to check (YYYY-MM-DD format)",
                    required=True,
                ),
                FunctionParameter(
                    name="duration_minutes",
                    type=ParameterType.INTEGER,
                    description="Duration of appointment in minutes",
                    required=False,
                    default=30,
                ),
            ],
            tags=["calendar", "scheduling"],
        )

        self.registry.register(
            name="book_appointment",
            description="Book an appointment at a specific date and time. Requires confirmation.",
            handler=self.book_appointment,
            parameters=[
                FunctionParameter(
                    name="date",
                    type=ParameterType.STRING,
                    description="Date of appointment (YYYY-MM-DD)",
                    required=True,
                ),
                FunctionParameter(
                    name="time",
                    type=ParameterType.STRING,
                    description="Time of appointment (HH:MM)",
                    required=True,
                ),
                FunctionParameter(
                    name="duration_minutes",
                    type=ParameterType.INTEGER,
                    description="Duration in minutes",
                    required=False,
                    default=30,
                ),
                FunctionParameter(
                    name="customer_name",
                    type=ParameterType.STRING,
                    description="Customer name",
                    required=True,
                ),
                FunctionParameter(
                    name="customer_phone",
                    type=ParameterType.STRING,
                    description="Customer phone number",
                    required=False,
                ),
                FunctionParameter(
                    name="notes",
                    type=ParameterType.STRING,
                    description="Additional notes",
                    required=False,
                ),
            ],
            requires_confirmation=True,
            tags=["calendar", "scheduling", "booking"],
        )

        self.registry.register(
            name="cancel_appointment",
            description="Cancel an existing appointment.",
            handler=self.cancel_appointment,
            parameters=[
                FunctionParameter(
                    name="appointment_id",
                    type=ParameterType.STRING,
                    description="Appointment ID to cancel",
                    required=True,
                ),
                FunctionParameter(
                    name="reason",
                    type=ParameterType.STRING,
                    description="Cancellation reason",
                    required=False,
                ),
            ],
            requires_confirmation=True,
            tags=["calendar", "scheduling"],
        )

        # =====================================================================
        # Information Functions
        # =====================================================================

        self.registry.register(
            name="get_business_hours",
            description="Get business operating hours.",
            handler=self.get_business_hours,
            parameters=[
                FunctionParameter(
                    name="day",
                    type=ParameterType.STRING,
                    description="Day of week (monday, tuesday, etc.) or 'today'",
                    required=False,
                    default="today",
                ),
            ],
            tags=["information", "business"],
        )

        self.registry.register(
            name="get_services",
            description="Get list of available services with prices.",
            handler=self.get_services,
            parameters=[
                FunctionParameter(
                    name="category",
                    type=ParameterType.STRING,
                    description="Service category filter",
                    required=False,
                ),
            ],
            tags=["information", "services"],
        )

        self.registry.register(
            name="get_location",
            description="Get business location and directions.",
            handler=self.get_location,
            parameters=[],
            tags=["information", "location"],
        )

        # =====================================================================
        # Call Control Functions
        # =====================================================================

        self.registry.register(
            name="transfer_call",
            description="Transfer the call to a human agent or specific department.",
            handler=self.transfer_call,
            parameters=[
                FunctionParameter(
                    name="department",
                    type=ParameterType.STRING,
                    description="Department to transfer to",
                    required=True,
                    enum=["sales", "support", "billing", "manager"],
                ),
                FunctionParameter(
                    name="reason",
                    type=ParameterType.STRING,
                    description="Reason for transfer",
                    required=False,
                ),
            ],
            tags=["call_control"],
        )

        self.registry.register(
            name="end_call",
            description="End the current call politely.",
            handler=self.end_call,
            parameters=[
                FunctionParameter(
                    name="reason",
                    type=ParameterType.STRING,
                    description="Reason for ending call",
                    required=False,
                    default="completed",
                ),
            ],
            tags=["call_control"],
        )

        # =====================================================================
        # Data Collection Functions
        # =====================================================================

        self.registry.register(
            name="collect_customer_info",
            description="Collect and store customer information.",
            handler=self.collect_customer_info,
            parameters=[
                FunctionParameter(
                    name="name",
                    type=ParameterType.STRING,
                    description="Customer full name",
                    required=True,
                ),
                FunctionParameter(
                    name="phone",
                    type=ParameterType.STRING,
                    description="Customer phone number",
                    required=False,
                ),
                FunctionParameter(
                    name="email",
                    type=ParameterType.STRING,
                    description="Customer email address",
                    required=False,
                ),
                FunctionParameter(
                    name="notes",
                    type=ParameterType.STRING,
                    description="Additional notes",
                    required=False,
                ),
            ],
            tags=["data", "customer"],
        )

        self.registry.register(
            name="log_issue",
            description="Log a customer issue or complaint for follow-up.",
            handler=self.log_issue,
            parameters=[
                FunctionParameter(
                    name="category",
                    type=ParameterType.STRING,
                    description="Issue category",
                    required=True,
                    enum=["billing", "service", "product", "other"],
                ),
                FunctionParameter(
                    name="description",
                    type=ParameterType.STRING,
                    description="Issue description",
                    required=True,
                ),
                FunctionParameter(
                    name="priority",
                    type=ParameterType.STRING,
                    description="Issue priority",
                    required=False,
                    default="medium",
                    enum=["low", "medium", "high", "urgent"],
                ),
            ],
            tags=["data", "support"],
        )

    # =========================================================================
    # Function Implementations
    # =========================================================================

    async def check_availability(
        self,
        date: str,
        duration_minutes: int = 30,
        **kwargs,
    ) -> Dict[str, Any]:
        """Check calendar availability."""
        # In production: integrate with actual calendar system
        # For now: return mock availability

        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD."}

        # Generate mock slots
        slots = []
        base_time = target_date.replace(hour=9, minute=0)

        for i in range(16):  # 9 AM to 5 PM
            slot_time = base_time + timedelta(minutes=i * 30)
            if slot_time.hour < 17:  # Before 5 PM
                # Mock: mark some as unavailable
                if i not in [3, 7, 8, 12]:
                    slots.append(slot_time.strftime("%H:%M"))

        return {
            "date": date,
            "available_slots": slots,
            "duration_minutes": duration_minutes,
        }

    async def book_appointment(
        self,
        date: str,
        time: str,
        customer_name: str,
        duration_minutes: int = 30,
        customer_phone: Optional[str] = None,
        notes: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Book an appointment."""
        import uuid

        # In production: integrate with actual booking system

        appointment_id = str(uuid.uuid4())[:8]

        logger.info(
            "appointment_booked",
            appointment_id=appointment_id,
            date=date,
            time=time,
            customer=customer_name,
        )

        return {
            "success": True,
            "appointment_id": appointment_id,
            "date": date,
            "time": time,
            "duration_minutes": duration_minutes,
            "customer_name": customer_name,
            "confirmation_message": f"Your appointment has been booked for {date} at {time}. Your confirmation number is {appointment_id}.",
        }

    async def cancel_appointment(
        self,
        appointment_id: str,
        reason: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Cancel an appointment."""
        # In production: integrate with actual booking system

        logger.info(
            "appointment_cancelled",
            appointment_id=appointment_id,
            reason=reason,
        )

        return {
            "success": True,
            "appointment_id": appointment_id,
            "message": f"Appointment {appointment_id} has been cancelled.",
        }

    async def get_business_hours(
        self,
        day: str = "today",
        **kwargs,
    ) -> Dict[str, Any]:
        """Get business hours."""
        hours = {
            "monday": {"open": "09:00", "close": "17:00"},
            "tuesday": {"open": "09:00", "close": "17:00"},
            "wednesday": {"open": "09:00", "close": "17:00"},
            "thursday": {"open": "09:00", "close": "17:00"},
            "friday": {"open": "09:00", "close": "17:00"},
            "saturday": {"open": "10:00", "close": "14:00"},
            "sunday": {"open": None, "close": None, "status": "closed"},
        }

        if day == "today":
            day = datetime.now().strftime("%A").lower()

        if day in hours:
            return {
                "day": day,
                **hours[day],
            }

        return {"all_hours": hours}

    async def get_services(
        self,
        category: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get available services."""
        services = [
            {"name": "Consultation", "duration": 30, "price": 50, "category": "general"},
            {"name": "Follow-up", "duration": 15, "price": 25, "category": "general"},
            {"name": "Full Assessment", "duration": 60, "price": 100, "category": "assessment"},
            {"name": "Emergency", "duration": 30, "price": 150, "category": "urgent"},
        ]

        if category:
            services = [s for s in services if s["category"] == category]

        return {"services": services}

    async def get_location(self, **kwargs) -> Dict[str, Any]:
        """Get business location."""
        return {
            "address": "123 Main Street, Suite 100",
            "city": "San Francisco",
            "state": "CA",
            "zip": "94102",
            "phone": "(555) 123-4567",
            "directions": "Located on the corner of Main and First, across from Central Park. Parking available in the building garage.",
        }

    async def transfer_call(
        self,
        department: str,
        reason: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Transfer call to department."""
        # In production: trigger actual call transfer

        context = kwargs.get("_context", {})
        session_id = context.get("session_id", "unknown")

        logger.info(
            "call_transfer_requested",
            session_id=session_id,
            department=department,
            reason=reason,
        )

        return {
            "action": "transfer",
            "department": department,
            "message": f"Transferring you to {department}. Please hold.",
        }

    async def end_call(
        self,
        reason: str = "completed",
        **kwargs,
    ) -> Dict[str, Any]:
        """End the current call."""
        context = kwargs.get("_context", {})
        session_id = context.get("session_id", "unknown")

        logger.info(
            "call_end_requested",
            session_id=session_id,
            reason=reason,
        )

        return {
            "action": "end_call",
            "reason": reason,
            "message": "Thank you for calling. Goodbye!",
        }

    async def collect_customer_info(
        self,
        name: str,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        notes: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Collect customer information."""
        # In production: store in CRM/database

        context = kwargs.get("_context", {})

        logger.info(
            "customer_info_collected",
            name=name,
            has_phone=phone is not None,
            has_email=email is not None,
        )

        return {
            "success": True,
            "message": "Customer information recorded.",
            "data": {
                "name": name,
                "phone": phone,
                "email": email,
                "notes": notes,
            },
        }

    async def log_issue(
        self,
        category: str,
        description: str,
        priority: str = "medium",
        **kwargs,
    ) -> Dict[str, Any]:
        """Log a customer issue."""
        import uuid

        ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"

        logger.info(
            "issue_logged",
            ticket_id=ticket_id,
            category=category,
            priority=priority,
        )

        return {
            "success": True,
            "ticket_id": ticket_id,
            "category": category,
            "priority": priority,
            "message": f"Issue logged with ticket number {ticket_id}. Our team will follow up within 24 hours.",
        }


# Initialize built-in functions on import
builtin_functions = BuiltinFunctions()

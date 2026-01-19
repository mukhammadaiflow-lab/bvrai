"""Calendar integrations."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date, time
from abc import abstractmethod
import logging
import urllib.parse

from app.integrations.base import (
    Integration,
    IntegrationConfig,
    IntegrationStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class CalendarEvent:
    """Calendar event."""
    id: str
    title: str
    start: datetime
    end: datetime
    description: Optional[str] = None
    location: Optional[str] = None
    attendees: List[str] = field(default_factory=list)
    is_all_day: bool = False
    recurring: bool = False
    calendar_id: Optional[str] = None
    meeting_link: Optional[str] = None
    status: str = "confirmed"  # confirmed, tentative, cancelled

    @property
    def duration_minutes(self) -> int:
        """Get event duration in minutes."""
        return int((self.end - self.start).total_seconds() / 60)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "description": self.description,
            "location": self.location,
            "attendees": self.attendees,
            "is_all_day": self.is_all_day,
            "recurring": self.recurring,
            "duration_minutes": self.duration_minutes,
            "meeting_link": self.meeting_link,
            "status": self.status,
        }


@dataclass
class CalendarSlot:
    """Available time slot."""
    start: datetime
    end: datetime
    calendar_id: Optional[str] = None

    @property
    def duration_minutes(self) -> int:
        """Get slot duration in minutes."""
        return int((self.end - self.start).total_seconds() / 60)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_minutes": self.duration_minutes,
        }


class CalendarIntegration(Integration):
    """Base class for calendar integrations."""

    @abstractmethod
    async def list_events(
        self,
        start: datetime,
        end: datetime,
        calendar_id: Optional[str] = None,
    ) -> List[CalendarEvent]:
        """List events in date range."""
        pass

    @abstractmethod
    async def get_event(self, event_id: str, calendar_id: Optional[str] = None) -> Optional[CalendarEvent]:
        """Get a specific event."""
        pass

    @abstractmethod
    async def create_event(self, event: CalendarEvent) -> CalendarEvent:
        """Create a new event."""
        pass

    @abstractmethod
    async def update_event(self, event: CalendarEvent) -> CalendarEvent:
        """Update an existing event."""
        pass

    @abstractmethod
    async def delete_event(self, event_id: str, calendar_id: Optional[str] = None) -> bool:
        """Delete an event."""
        pass

    @abstractmethod
    async def get_free_busy(
        self,
        start: datetime,
        end: datetime,
        calendar_id: Optional[str] = None,
    ) -> List[tuple]:
        """Get busy times (list of (start, end) tuples)."""
        pass

    async def find_available_slots(
        self,
        start: datetime,
        end: datetime,
        duration_minutes: int = 30,
        calendar_id: Optional[str] = None,
        working_hours: Optional[tuple] = None,
    ) -> List[CalendarSlot]:
        """
        Find available time slots.

        Args:
            start: Start of search range
            end: End of search range
            duration_minutes: Required slot duration
            calendar_id: Calendar to check
            working_hours: Tuple of (start_hour, end_hour) for working hours

        Returns:
            List of available slots
        """
        working_hours = working_hours or (9, 17)  # 9 AM to 5 PM default

        # Get busy times
        busy_times = await self.get_free_busy(start, end, calendar_id)

        # Sort busy times
        busy_times = sorted(busy_times, key=lambda x: x[0])

        slots = []
        current = start

        # Iterate through each day
        while current < end:
            day_start = current.replace(
                hour=working_hours[0], minute=0, second=0, microsecond=0
            )
            day_end = current.replace(
                hour=working_hours[1], minute=0, second=0, microsecond=0
            )

            # Skip if before working hours
            if current < day_start:
                current = day_start

            # Find slots for this day
            slot_start = current

            for busy_start, busy_end in busy_times:
                if busy_end <= slot_start:
                    continue
                if busy_start >= day_end:
                    break

                # Check if there's a slot before this busy time
                if busy_start > slot_start:
                    slot_end = min(busy_start, day_end)
                    if (slot_end - slot_start).total_seconds() / 60 >= duration_minutes:
                        slots.append(CalendarSlot(
                            start=slot_start,
                            end=slot_end,
                            calendar_id=calendar_id,
                        ))

                slot_start = max(slot_start, busy_end)

            # Check for slot after last busy time
            if slot_start < day_end:
                if (day_end - slot_start).total_seconds() / 60 >= duration_minutes:
                    slots.append(CalendarSlot(
                        start=slot_start,
                        end=day_end,
                        calendar_id=calendar_id,
                    ))

            # Move to next day
            current = (current + timedelta(days=1)).replace(
                hour=working_hours[0], minute=0, second=0, microsecond=0
            )

        return slots

    async def schedule_meeting(
        self,
        title: str,
        start: datetime,
        duration_minutes: int,
        attendees: Optional[List[str]] = None,
        description: Optional[str] = None,
        calendar_id: Optional[str] = None,
    ) -> CalendarEvent:
        """Schedule a meeting."""
        event = CalendarEvent(
            id="",
            title=title,
            start=start,
            end=start + timedelta(minutes=duration_minutes),
            description=description,
            attendees=attendees or [],
            calendar_id=calendar_id,
        )
        return await self.create_event(event)


class GoogleCalendarIntegration(CalendarIntegration):
    """
    Google Calendar integration.

    Usage:
        integration = GoogleCalendarIntegration(config)
        await integration.connect(credentials)

        events = await integration.list_events(start, end)
    """

    PROVIDER_NAME = "google_calendar"
    REQUIRED_SCOPES = [
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/calendar.events",
    ]

    BASE_URL = "https://www.googleapis.com/calendar/v3"

    def get_oauth_url(self, redirect_uri: str, state: str) -> str:
        """Get Google OAuth URL."""
        client_id = self.config.settings.get("client_id", "")
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": state,
            "scope": " ".join(self.REQUIRED_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
        }
        return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange code for tokens."""
        client = await self._get_http_client()
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": self.config.settings.get("client_id"),
                "client_secret": self.config.settings.get("client_secret"),
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        response.raise_for_status()
        return response.json()

    async def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to Google Calendar."""
        self.config.credentials = credentials
        self.config.status = IntegrationStatus.CONNECTED
        self.config.connected_at = datetime.utcnow()
        return True

    async def disconnect(self) -> bool:
        """Disconnect from Google Calendar."""
        self.config.status = IntegrationStatus.DISCONNECTED
        self.config.credentials = {}
        return True

    async def test_connection(self) -> bool:
        """Test Google Calendar connection."""
        try:
            await self._make_request("GET", f"{self.BASE_URL}/users/me/calendarList")
            return True
        except Exception as e:
            logger.error(f"Google Calendar connection test failed: {e}")
            return False

    async def refresh_token(self) -> bool:
        """Refresh Google token."""
        try:
            client = await self._get_http_client()
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": self.config.settings.get("client_id"),
                    "client_secret": self.config.settings.get("client_secret"),
                    "refresh_token": self.config.credentials.get("refresh_token"),
                    "grant_type": "refresh_token",
                },
            )
            response.raise_for_status()
            data = response.json()
            self.config.credentials["access_token"] = data["access_token"]
            return True
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False

    async def list_events(
        self,
        start: datetime,
        end: datetime,
        calendar_id: Optional[str] = None,
    ) -> List[CalendarEvent]:
        """List events from Google Calendar."""
        calendar_id = calendar_id or "primary"
        try:
            data = await self._make_request(
                "GET",
                f"{self.BASE_URL}/calendars/{calendar_id}/events",
                params={
                    "timeMin": start.isoformat() + "Z",
                    "timeMax": end.isoformat() + "Z",
                    "singleEvents": "true",
                    "orderBy": "startTime",
                },
            )
            return [self._parse_event(e) for e in data.get("items", [])]
        except Exception as e:
            logger.error(f"Failed to list events: {e}")
            return []

    async def get_event(self, event_id: str, calendar_id: Optional[str] = None) -> Optional[CalendarEvent]:
        """Get event from Google Calendar."""
        calendar_id = calendar_id or "primary"
        try:
            data = await self._make_request(
                "GET",
                f"{self.BASE_URL}/calendars/{calendar_id}/events/{event_id}",
            )
            return self._parse_event(data)
        except Exception as e:
            logger.error(f"Failed to get event: {e}")
            return None

    async def create_event(self, event: CalendarEvent) -> CalendarEvent:
        """Create event in Google Calendar."""
        calendar_id = event.calendar_id or "primary"

        body = {
            "summary": event.title,
            "description": event.description,
            "location": event.location,
            "start": {"dateTime": event.start.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": event.end.isoformat(), "timeZone": "UTC"},
        }

        if event.attendees:
            body["attendees"] = [{"email": email} for email in event.attendees]

        data = await self._make_request(
            "POST",
            f"{self.BASE_URL}/calendars/{calendar_id}/events",
            json=body,
        )

        return self._parse_event(data)

    async def update_event(self, event: CalendarEvent) -> CalendarEvent:
        """Update event in Google Calendar."""
        calendar_id = event.calendar_id or "primary"

        body = {
            "summary": event.title,
            "description": event.description,
            "location": event.location,
            "start": {"dateTime": event.start.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": event.end.isoformat(), "timeZone": "UTC"},
        }

        if event.attendees:
            body["attendees"] = [{"email": email} for email in event.attendees]

        data = await self._make_request(
            "PUT",
            f"{self.BASE_URL}/calendars/{calendar_id}/events/{event.id}",
            json=body,
        )

        return self._parse_event(data)

    async def delete_event(self, event_id: str, calendar_id: Optional[str] = None) -> bool:
        """Delete event from Google Calendar."""
        calendar_id = calendar_id or "primary"
        try:
            client = await self._get_http_client()
            headers = {"Authorization": f"Bearer {self.config.credentials.get('access_token')}"}
            response = await client.delete(
                f"{self.BASE_URL}/calendars/{calendar_id}/events/{event_id}",
                headers=headers,
            )
            return response.status_code == 204
        except Exception as e:
            logger.error(f"Failed to delete event: {e}")
            return False

    async def get_free_busy(
        self,
        start: datetime,
        end: datetime,
        calendar_id: Optional[str] = None,
    ) -> List[tuple]:
        """Get busy times from Google Calendar."""
        calendar_id = calendar_id or "primary"
        try:
            data = await self._make_request(
                "POST",
                f"{self.BASE_URL}/freeBusy",
                json={
                    "timeMin": start.isoformat() + "Z",
                    "timeMax": end.isoformat() + "Z",
                    "items": [{"id": calendar_id}],
                },
            )

            busy_times = []
            calendars = data.get("calendars", {})
            for cal_data in calendars.values():
                for busy in cal_data.get("busy", []):
                    busy_start = datetime.fromisoformat(busy["start"].replace("Z", "+00:00"))
                    busy_end = datetime.fromisoformat(busy["end"].replace("Z", "+00:00"))
                    busy_times.append((busy_start.replace(tzinfo=None), busy_end.replace(tzinfo=None)))

            return busy_times
        except Exception as e:
            logger.error(f"Failed to get free/busy: {e}")
            return []

    def _parse_event(self, data: Dict[str, Any]) -> CalendarEvent:
        """Parse Google Calendar event."""
        start_data = data.get("start", {})
        end_data = data.get("end", {})

        # Handle all-day events
        if "date" in start_data:
            start = datetime.strptime(start_data["date"], "%Y-%m-%d")
            end = datetime.strptime(end_data["date"], "%Y-%m-%d")
            is_all_day = True
        else:
            start = datetime.fromisoformat(start_data.get("dateTime", "").replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_data.get("dateTime", "").replace("Z", "+00:00"))
            is_all_day = False

        attendees = [a.get("email") for a in data.get("attendees", [])]

        return CalendarEvent(
            id=data.get("id", ""),
            title=data.get("summary", ""),
            start=start.replace(tzinfo=None),
            end=end.replace(tzinfo=None),
            description=data.get("description"),
            location=data.get("location"),
            attendees=attendees,
            is_all_day=is_all_day,
            recurring=data.get("recurringEventId") is not None,
            meeting_link=data.get("hangoutLink"),
            status=data.get("status", "confirmed"),
        )


class OutlookCalendarIntegration(CalendarIntegration):
    """
    Microsoft Outlook/365 Calendar integration.

    Usage:
        integration = OutlookCalendarIntegration(config)
        await integration.connect(credentials)

        events = await integration.list_events(start, end)
    """

    PROVIDER_NAME = "outlook_calendar"
    REQUIRED_SCOPES = ["Calendars.ReadWrite", "User.Read"]

    BASE_URL = "https://graph.microsoft.com/v1.0"

    def get_oauth_url(self, redirect_uri: str, state: str) -> str:
        """Get Microsoft OAuth URL."""
        client_id = self.config.settings.get("client_id", "")
        tenant = self.config.settings.get("tenant", "common")
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": state,
            "scope": " ".join(self.REQUIRED_SCOPES),
            "response_mode": "query",
        }
        return f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize?{urllib.parse.urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange code for tokens."""
        client = await self._get_http_client()
        tenant = self.config.settings.get("tenant", "common")
        response = await client.post(
            f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
            data={
                "code": code,
                "client_id": self.config.settings.get("client_id"),
                "client_secret": self.config.settings.get("client_secret"),
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
                "scope": " ".join(self.REQUIRED_SCOPES),
            },
        )
        response.raise_for_status()
        return response.json()

    async def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to Outlook Calendar."""
        self.config.credentials = credentials
        self.config.status = IntegrationStatus.CONNECTED
        self.config.connected_at = datetime.utcnow()
        return True

    async def disconnect(self) -> bool:
        """Disconnect from Outlook Calendar."""
        self.config.status = IntegrationStatus.DISCONNECTED
        self.config.credentials = {}
        return True

    async def test_connection(self) -> bool:
        """Test Outlook Calendar connection."""
        try:
            await self._make_request("GET", f"{self.BASE_URL}/me/calendars")
            return True
        except Exception as e:
            logger.error(f"Outlook Calendar connection test failed: {e}")
            return False

    async def refresh_token(self) -> bool:
        """Refresh Microsoft token."""
        try:
            client = await self._get_http_client()
            tenant = self.config.settings.get("tenant", "common")
            response = await client.post(
                f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
                data={
                    "client_id": self.config.settings.get("client_id"),
                    "client_secret": self.config.settings.get("client_secret"),
                    "refresh_token": self.config.credentials.get("refresh_token"),
                    "grant_type": "refresh_token",
                    "scope": " ".join(self.REQUIRED_SCOPES),
                },
            )
            response.raise_for_status()
            data = response.json()
            self.config.credentials["access_token"] = data["access_token"]
            self.config.credentials["refresh_token"] = data.get("refresh_token", self.config.credentials.get("refresh_token"))
            return True
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False

    async def list_events(
        self,
        start: datetime,
        end: datetime,
        calendar_id: Optional[str] = None,
    ) -> List[CalendarEvent]:
        """List events from Outlook Calendar."""
        try:
            params = {
                "startDateTime": start.isoformat() + "Z",
                "endDateTime": end.isoformat() + "Z",
                "$orderby": "start/dateTime",
            }
            endpoint = f"{self.BASE_URL}/me/calendar/calendarView" if not calendar_id else f"{self.BASE_URL}/me/calendars/{calendar_id}/calendarView"
            data = await self._make_request("GET", endpoint, params=params)
            return [self._parse_event(e) for e in data.get("value", [])]
        except Exception as e:
            logger.error(f"Failed to list events: {e}")
            return []

    async def get_event(self, event_id: str, calendar_id: Optional[str] = None) -> Optional[CalendarEvent]:
        """Get event from Outlook Calendar."""
        try:
            endpoint = f"{self.BASE_URL}/me/calendar/events/{event_id}" if not calendar_id else f"{self.BASE_URL}/me/calendars/{calendar_id}/events/{event_id}"
            data = await self._make_request("GET", endpoint)
            return self._parse_event(data)
        except Exception as e:
            logger.error(f"Failed to get event: {e}")
            return None

    async def create_event(self, event: CalendarEvent) -> CalendarEvent:
        """Create event in Outlook Calendar."""
        body = {
            "subject": event.title,
            "body": {"contentType": "text", "content": event.description or ""},
            "start": {"dateTime": event.start.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": event.end.isoformat(), "timeZone": "UTC"},
            "location": {"displayName": event.location} if event.location else None,
        }

        if event.attendees:
            body["attendees"] = [
                {"emailAddress": {"address": email}, "type": "required"}
                for email in event.attendees
            ]

        endpoint = f"{self.BASE_URL}/me/calendar/events" if not event.calendar_id else f"{self.BASE_URL}/me/calendars/{event.calendar_id}/events"
        data = await self._make_request("POST", endpoint, json=body)

        return self._parse_event(data)

    async def update_event(self, event: CalendarEvent) -> CalendarEvent:
        """Update event in Outlook Calendar."""
        body = {
            "subject": event.title,
            "body": {"contentType": "text", "content": event.description or ""},
            "start": {"dateTime": event.start.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": event.end.isoformat(), "timeZone": "UTC"},
        }

        if event.location:
            body["location"] = {"displayName": event.location}

        endpoint = f"{self.BASE_URL}/me/calendar/events/{event.id}" if not event.calendar_id else f"{self.BASE_URL}/me/calendars/{event.calendar_id}/events/{event.id}"
        data = await self._make_request("PATCH", endpoint, json=body)

        return self._parse_event(data)

    async def delete_event(self, event_id: str, calendar_id: Optional[str] = None) -> bool:
        """Delete event from Outlook Calendar."""
        try:
            client = await self._get_http_client()
            headers = {"Authorization": f"Bearer {self.config.credentials.get('access_token')}"}
            endpoint = f"{self.BASE_URL}/me/calendar/events/{event_id}" if not calendar_id else f"{self.BASE_URL}/me/calendars/{calendar_id}/events/{event_id}"
            response = await client.delete(endpoint, headers=headers)
            return response.status_code == 204
        except Exception as e:
            logger.error(f"Failed to delete event: {e}")
            return False

    async def get_free_busy(
        self,
        start: datetime,
        end: datetime,
        calendar_id: Optional[str] = None,
    ) -> List[tuple]:
        """Get busy times from Outlook Calendar."""
        # Outlook uses events to determine busy times
        events = await self.list_events(start, end, calendar_id)
        return [(e.start, e.end) for e in events if e.status != "cancelled"]

    def _parse_event(self, data: Dict[str, Any]) -> CalendarEvent:
        """Parse Outlook Calendar event."""
        start_data = data.get("start", {})
        end_data = data.get("end", {})

        start = datetime.fromisoformat(start_data.get("dateTime", "").replace("Z", ""))
        end = datetime.fromisoformat(end_data.get("dateTime", "").replace("Z", ""))

        attendees = [
            a.get("emailAddress", {}).get("address")
            for a in data.get("attendees", [])
        ]

        return CalendarEvent(
            id=data.get("id", ""),
            title=data.get("subject", ""),
            start=start,
            end=end,
            description=data.get("body", {}).get("content"),
            location=data.get("location", {}).get("displayName"),
            attendees=[a for a in attendees if a],
            is_all_day=data.get("isAllDay", False),
            recurring=data.get("recurrence") is not None,
            meeting_link=data.get("onlineMeeting", {}).get("joinUrl"),
            status="confirmed" if not data.get("isCancelled") else "cancelled",
        )

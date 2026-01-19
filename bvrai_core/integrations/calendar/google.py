"""
Google Calendar Integration

This module provides Google Calendar integration for the voice agent platform,
enabling calendar synchronization, availability checking, and meeting scheduling.

Features:
- OAuth2 authentication with token refresh
- Calendar listing and event management
- Free/busy availability queries
- Meeting scheduling with conferencing
- Recurring event support
- Calendar sharing and permissions
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp

from ..base import (
    CalendarProvider,
    CalendarEvent,
    Calendar,
    AuthType,
    IntegrationStatus,
    IntegrationError,
    AuthenticationError,
    RateLimitError,
    SyncState,
)


logger = logging.getLogger(__name__)


class GoogleCalendarProvider(CalendarProvider):
    """
    Google Calendar integration provider.

    Implements full Google Calendar API v3 functionality including:
    - Calendar and event CRUD operations
    - Free/busy availability queries
    - Meeting scheduling with Google Meet
    - Incremental sync with sync tokens
    - Watch notifications for real-time updates

    OAuth2 Scopes Required:
    - https://www.googleapis.com/auth/calendar (full access)
    - https://www.googleapis.com/auth/calendar.events (events only)
    - https://www.googleapis.com/auth/calendar.readonly (read only)
    """

    PROVIDER_NAME = "google_calendar"
    AUTH_TYPE = AuthType.OAUTH2

    # Google API endpoints
    BASE_URL = "https://www.googleapis.com/calendar/v3"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"

    # API configuration
    DEFAULT_MAX_RESULTS = 250
    MAX_ATTENDEES = 100
    RATE_LIMIT_DELAY = 1.0
    MAX_RETRIES = 3

    # Event colors (Google's predefined palette)
    EVENT_COLORS = {
        "1": "lavender",
        "2": "sage",
        "3": "grape",
        "4": "flamingo",
        "5": "banana",
        "6": "tangerine",
        "7": "peacock",
        "8": "graphite",
        "9": "blueberry",
        "10": "basil",
        "11": "tomato",
    }

    def __init__(
        self,
        integration_id: str,
        organization_id: str,
        credentials: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Google Calendar provider.

        Args:
            integration_id: Unique integration identifier
            organization_id: Organization this integration belongs to
            credentials: OAuth2 credentials including:
                - access_token: Current access token
                - refresh_token: Token for refreshing access
                - client_id: OAuth2 client ID
                - client_secret: OAuth2 client secret
                - token_expiry: Token expiration timestamp
            settings: Optional settings including:
                - default_calendar_id: Default calendar for operations
                - time_zone: Default time zone (e.g., "America/New_York")
                - default_reminder_minutes: Default reminder time
                - auto_add_meet: Automatically add Google Meet to events
        """
        super().__init__(integration_id, organization_id, credentials, settings or {})

        self._access_token = credentials.get("access_token")
        self._refresh_token = credentials.get("refresh_token")
        self._client_id = credentials.get("client_id")
        self._client_secret = credentials.get("client_secret")
        self._token_expiry = credentials.get("token_expiry")

        # Settings
        self._default_calendar_id = self.settings.get("default_calendar_id", "primary")
        self._time_zone = self.settings.get("time_zone", "UTC")
        self._default_reminder_minutes = self.settings.get("default_reminder_minutes", 30)
        self._auto_add_meet = self.settings.get("auto_add_meet", False)

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        # Sync tokens for incremental sync
        self._sync_tokens: Dict[str, str] = {}

        # Rate limiting
        self._last_request_time: Optional[datetime] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def _ensure_valid_token(self) -> None:
        """Ensure access token is valid, refreshing if needed."""
        if self._token_expiry:
            if isinstance(self._token_expiry, str):
                expiry = datetime.fromisoformat(self._token_expiry.replace("Z", "+00:00"))
            else:
                expiry = self._token_expiry

            # Refresh if token expires in less than 5 minutes
            if datetime.utcnow() >= expiry - timedelta(minutes=5):
                await self._refresh_access_token()

    async def _refresh_access_token(self) -> None:
        """Refresh the OAuth2 access token."""
        if not self._refresh_token:
            raise AuthenticationError("No refresh token available")

        if not self._client_id or not self._client_secret:
            raise AuthenticationError("Missing OAuth2 client credentials")

        session = await self._get_session()

        try:
            async with session.post(
                self.TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
            ) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise AuthenticationError(
                        f"Token refresh failed: {error_data.get('error_description', 'Unknown error')}"
                    )

                data = await response.json()
                self._access_token = data["access_token"]
                self._token_expiry = datetime.utcnow() + timedelta(
                    seconds=data.get("expires_in", 3600)
                )

                # Update credentials
                self.credentials["access_token"] = self._access_token
                self.credentials["token_expiry"] = self._token_expiry.isoformat()

                # Emit refresh event
                if self._on_token_refresh:
                    await self._on_token_refresh(
                        self.integration_id,
                        {
                            "access_token": self._access_token,
                            "token_expiry": self._token_expiry.isoformat(),
                        },
                    )

                logger.info(f"Refreshed access token for integration {self.integration_id}")

        except aiohttp.ClientError as e:
            raise IntegrationError(f"Network error during token refresh: {e}")

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to the Google Calendar API.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE)
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON request body
            retry_count: Current retry attempt

        Returns:
            JSON response data

        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limited
            IntegrationError: For other errors
        """
        await self._ensure_valid_token()

        # Rate limiting
        if self._last_request_time:
            elapsed = (datetime.utcnow() - self._last_request_time).total_seconds()
            if elapsed < self.RATE_LIMIT_DELAY:
                await asyncio.sleep(self.RATE_LIMIT_DELAY - elapsed)

        session = await self._get_session()
        url = f"{self.BASE_URL}/{endpoint}"

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        try:
            self._last_request_time = datetime.utcnow()

            async with session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_data,
            ) as response:
                # Handle rate limiting
                if response.status == 429:
                    if retry_count < self.MAX_RETRIES:
                        retry_after = int(response.headers.get("Retry-After", "5"))
                        logger.warning(
                            f"Rate limited, retrying after {retry_after}s "
                            f"(attempt {retry_count + 1}/{self.MAX_RETRIES})"
                        )
                        await asyncio.sleep(retry_after)
                        return await self._request(
                            method, endpoint, params, json_data, retry_count + 1
                        )
                    raise RateLimitError("Google Calendar API rate limit exceeded")

                # Handle authentication errors
                if response.status == 401:
                    if retry_count < 1:
                        # Try refreshing token once
                        await self._refresh_access_token()
                        return await self._request(
                            method, endpoint, params, json_data, retry_count + 1
                        )
                    raise AuthenticationError("Google Calendar authentication failed")

                # Handle not found
                if response.status == 404:
                    return {}

                # Handle other errors
                if response.status >= 400:
                    try:
                        error_data = await response.json()
                        error_message = error_data.get("error", {}).get(
                            "message", "Unknown error"
                        )
                    except Exception:
                        error_message = await response.text()
                    raise IntegrationError(
                        f"Google Calendar API error ({response.status}): {error_message}"
                    )

                # Return empty dict for 204 No Content
                if response.status == 204:
                    return {}

                return await response.json()

        except aiohttp.ClientError as e:
            if retry_count < self.MAX_RETRIES:
                await asyncio.sleep(2 ** retry_count)
                return await self._request(
                    method, endpoint, params, json_data, retry_count + 1
                )
            raise IntegrationError(f"Network error: {e}")

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Establish connection and verify credentials.

        Returns:
            True if connection successful
        """
        try:
            # Verify by fetching calendar list
            response = await self._request("GET", "users/me/calendarList", {"maxResults": 1})
            self._status = IntegrationStatus.CONNECTED
            logger.info(f"Connected to Google Calendar for integration {self.integration_id}")
            return True

        except AuthenticationError:
            self._status = IntegrationStatus.AUTH_FAILED
            return False
        except Exception as e:
            self._status = IntegrationStatus.ERROR
            logger.error(f"Failed to connect to Google Calendar: {e}")
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect and clean up resources.

        Returns:
            True if disconnection successful
        """
        try:
            if self._session and not self._session.closed:
                await self._session.close()

            self._status = IntegrationStatus.DISCONNECTED
            self._sync_tokens.clear()
            logger.info(f"Disconnected from Google Calendar for integration {self.integration_id}")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from Google Calendar: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the integration.

        Returns:
            Health check results with status and metrics
        """
        health = {
            "provider": self.PROVIDER_NAME,
            "status": "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "details": {},
        }

        try:
            # Test API access
            start = datetime.utcnow()
            response = await self._request("GET", "users/me/calendarList", {"maxResults": 1})
            latency_ms = int((datetime.utcnow() - start).total_seconds() * 1000)

            health["status"] = "healthy"
            health["details"] = {
                "api_latency_ms": latency_ms,
                "calendars_accessible": response.get("items", []) is not None,
                "token_valid": True,
            }

        except AuthenticationError:
            health["status"] = "unhealthy"
            health["details"]["error"] = "Authentication failed"
            health["details"]["token_valid"] = False

        except RateLimitError:
            health["status"] = "degraded"
            health["details"]["error"] = "Rate limited"

        except Exception as e:
            health["status"] = "unhealthy"
            health["details"]["error"] = str(e)

        return health

    # =========================================================================
    # Calendar Operations
    # =========================================================================

    async def list_calendars(self) -> List[Calendar]:
        """
        List all accessible calendars.

        Returns:
            List of Calendar objects
        """
        calendars = []
        page_token = None

        while True:
            params = {
                "maxResults": self.DEFAULT_MAX_RESULTS,
                "showHidden": False,
            }
            if page_token:
                params["pageToken"] = page_token

            response = await self._request("GET", "users/me/calendarList", params)

            for item in response.get("items", []):
                calendars.append(self._convert_calendar(item))

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return calendars

    async def get_calendar(self, calendar_id: str) -> Optional[Calendar]:
        """
        Get a specific calendar by ID.

        Args:
            calendar_id: Calendar identifier

        Returns:
            Calendar object or None if not found
        """
        response = await self._request("GET", f"users/me/calendarList/{calendar_id}")
        if not response:
            return None
        return self._convert_calendar(response)

    async def create_calendar(
        self,
        name: str,
        description: str = "",
        time_zone: Optional[str] = None,
    ) -> Calendar:
        """
        Create a new calendar.

        Args:
            name: Calendar name
            description: Calendar description
            time_zone: Calendar time zone

        Returns:
            Created Calendar object
        """
        data = {
            "summary": name,
            "description": description,
            "timeZone": time_zone or self._time_zone,
        }

        response = await self._request("POST", "calendars", json_data=data)
        return self._convert_calendar(response)

    async def delete_calendar(self, calendar_id: str) -> bool:
        """
        Delete a calendar.

        Args:
            calendar_id: Calendar identifier

        Returns:
            True if deletion successful
        """
        await self._request("DELETE", f"calendars/{calendar_id}")
        return True

    def _convert_calendar(self, data: Dict[str, Any]) -> Calendar:
        """Convert Google Calendar API response to Calendar object."""
        return Calendar(
            id=data["id"],
            name=data.get("summary", ""),
            description=data.get("description", ""),
            time_zone=data.get("timeZone", "UTC"),
            color=data.get("backgroundColor"),
            is_primary=data.get("primary", False),
            access_role=data.get("accessRole", "reader"),
            metadata={
                "etag": data.get("etag"),
                "kind": data.get("kind"),
                "foregroundColor": data.get("foregroundColor"),
                "selected": data.get("selected", False),
                "hidden": data.get("hidden", False),
            },
        )

    # =========================================================================
    # Event Operations
    # =========================================================================

    async def list_events(
        self,
        calendar_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> List[CalendarEvent]:
        """
        List calendar events within a time range.

        Args:
            calendar_id: Calendar to query (default: primary)
            start_time: Filter events starting after this time
            end_time: Filter events ending before this time
            limit: Maximum number of events to return
            cursor: Pagination cursor

        Returns:
            List of CalendarEvent objects
        """
        cal_id = calendar_id or self._default_calendar_id
        events = []

        params = {
            "maxResults": min(limit, self.DEFAULT_MAX_RESULTS),
            "singleEvents": True,
            "orderBy": "startTime",
        }

        if start_time:
            params["timeMin"] = start_time.isoformat() + "Z"
        if end_time:
            params["timeMax"] = end_time.isoformat() + "Z"
        if cursor:
            params["pageToken"] = cursor

        response = await self._request("GET", f"calendars/{cal_id}/events", params)

        for item in response.get("items", []):
            event = self._convert_event(item, cal_id)
            if event:
                events.append(event)

        return events

    async def get_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
    ) -> Optional[CalendarEvent]:
        """
        Get a specific event by ID.

        Args:
            event_id: Event identifier
            calendar_id: Calendar containing the event

        Returns:
            CalendarEvent object or None if not found
        """
        cal_id = calendar_id or self._default_calendar_id
        response = await self._request("GET", f"calendars/{cal_id}/events/{event_id}")

        if not response:
            return None

        return self._convert_event(response, cal_id)

    async def create_event(self, event: CalendarEvent) -> CalendarEvent:
        """
        Create a new calendar event.

        Args:
            event: CalendarEvent to create

        Returns:
            Created CalendarEvent with ID populated
        """
        cal_id = event.calendar_id or self._default_calendar_id
        data = self._build_event_data(event)

        # Add Google Meet if configured
        if self._auto_add_meet and not event.meeting_link:
            data["conferenceData"] = {
                "createRequest": {
                    "requestId": f"meet-{event.id or hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]}",
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }

        params = {}
        if "conferenceData" in data:
            params["conferenceDataVersion"] = 1

        response = await self._request(
            "POST",
            f"calendars/{cal_id}/events",
            params=params,
            json_data=data,
        )

        return self._convert_event(response, cal_id)

    async def update_event(self, event: CalendarEvent) -> CalendarEvent:
        """
        Update an existing calendar event.

        Args:
            event: CalendarEvent with updated fields

        Returns:
            Updated CalendarEvent
        """
        if not event.id:
            raise IntegrationError("Event ID required for update")

        cal_id = event.calendar_id or self._default_calendar_id
        data = self._build_event_data(event)

        response = await self._request(
            "PUT",
            f"calendars/{cal_id}/events/{event.id}",
            json_data=data,
        )

        return self._convert_event(response, cal_id)

    async def delete_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
        send_updates: str = "all",
    ) -> bool:
        """
        Delete a calendar event.

        Args:
            event_id: Event identifier
            calendar_id: Calendar containing the event
            send_updates: Notification setting (all, externalOnly, none)

        Returns:
            True if deletion successful
        """
        cal_id = calendar_id or self._default_calendar_id
        await self._request(
            "DELETE",
            f"calendars/{cal_id}/events/{event_id}",
            params={"sendUpdates": send_updates},
        )
        return True

    def _build_event_data(self, event: CalendarEvent) -> Dict[str, Any]:
        """Build Google Calendar API event data from CalendarEvent."""
        data = {
            "summary": event.title,
            "description": event.description or "",
            "location": event.location or "",
        }

        # Handle all-day vs timed events
        if event.all_day:
            data["start"] = {"date": event.start_time.strftime("%Y-%m-%d")}
            data["end"] = {"date": event.end_time.strftime("%Y-%m-%d")}
        else:
            data["start"] = {
                "dateTime": event.start_time.isoformat(),
                "timeZone": event.time_zone or self._time_zone,
            }
            data["end"] = {
                "dateTime": event.end_time.isoformat(),
                "timeZone": event.time_zone or self._time_zone,
            }

        # Add attendees
        if event.attendees:
            data["attendees"] = [
                {
                    "email": attendee.get("email"),
                    "displayName": attendee.get("name"),
                    "optional": attendee.get("optional", False),
                    "responseStatus": attendee.get("status", "needsAction"),
                }
                for attendee in event.attendees[:self.MAX_ATTENDEES]
            ]

        # Add reminders
        if event.reminders:
            data["reminders"] = {
                "useDefault": False,
                "overrides": [
                    {"method": r.get("method", "popup"), "minutes": r.get("minutes", 30)}
                    for r in event.reminders
                ],
            }
        else:
            data["reminders"] = {
                "useDefault": False,
                "overrides": [
                    {"method": "popup", "minutes": self._default_reminder_minutes}
                ],
            }

        # Add recurrence
        if event.recurrence:
            data["recurrence"] = event.recurrence

        # Add visibility
        if event.visibility:
            data["visibility"] = event.visibility

        # Add status
        if event.status:
            data["status"] = event.status

        return data

    def _convert_event(
        self,
        data: Dict[str, Any],
        calendar_id: str,
    ) -> Optional[CalendarEvent]:
        """Convert Google Calendar API response to CalendarEvent object."""
        if not data:
            return None

        # Parse start/end times
        start_data = data.get("start", {})
        end_data = data.get("end", {})

        all_day = "date" in start_data

        if all_day:
            start_time = datetime.strptime(start_data["date"], "%Y-%m-%d")
            end_time = datetime.strptime(end_data["date"], "%Y-%m-%d")
        else:
            start_str = start_data.get("dateTime", "")
            end_str = end_data.get("dateTime", "")

            # Handle various datetime formats
            try:
                start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                end_time = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except ValueError:
                logger.warning(f"Could not parse event times: {start_str}, {end_str}")
                return None

        # Parse attendees
        attendees = []
        for attendee in data.get("attendees", []):
            attendees.append({
                "email": attendee.get("email"),
                "name": attendee.get("displayName"),
                "status": attendee.get("responseStatus"),
                "organizer": attendee.get("organizer", False),
                "optional": attendee.get("optional", False),
            })

        # Parse organizer
        organizer = None
        organizer_data = data.get("organizer")
        if organizer_data:
            organizer = {
                "email": organizer_data.get("email"),
                "name": organizer_data.get("displayName"),
            }

        # Parse reminders
        reminders = []
        reminder_data = data.get("reminders", {})
        if not reminder_data.get("useDefault"):
            for override in reminder_data.get("overrides", []):
                reminders.append({
                    "method": override.get("method"),
                    "minutes": override.get("minutes"),
                })

        # Get meeting link from conference data
        meeting_link = None
        conference_data = data.get("conferenceData")
        if conference_data:
            entry_points = conference_data.get("entryPoints", [])
            for entry_point in entry_points:
                if entry_point.get("entryPointType") == "video":
                    meeting_link = entry_point.get("uri")
                    break

        return CalendarEvent(
            id=data.get("id", ""),
            calendar_id=calendar_id,
            title=data.get("summary", "(No title)"),
            description=data.get("description"),
            location=data.get("location"),
            start_time=start_time,
            end_time=end_time,
            time_zone=start_data.get("timeZone") or end_data.get("timeZone"),
            all_day=all_day,
            attendees=attendees,
            organizer=organizer,
            reminders=reminders,
            recurrence=data.get("recurrence"),
            meeting_link=meeting_link,
            status=data.get("status"),
            visibility=data.get("visibility"),
            created_at=datetime.fromisoformat(
                data.get("created", datetime.utcnow().isoformat()).replace("Z", "+00:00")
            ) if data.get("created") else None,
            updated_at=datetime.fromisoformat(
                data.get("updated", datetime.utcnow().isoformat()).replace("Z", "+00:00")
            ) if data.get("updated") else None,
            metadata={
                "etag": data.get("etag"),
                "htmlLink": data.get("htmlLink"),
                "iCalUID": data.get("iCalUID"),
                "recurringEventId": data.get("recurringEventId"),
                "colorId": data.get("colorId"),
                "creator": data.get("creator"),
            },
        )

    # =========================================================================
    # Availability Operations
    # =========================================================================

    async def get_availability(
        self,
        calendar_ids: List[str],
        start_time: datetime,
        end_time: datetime,
        slot_duration_minutes: int = 30,
    ) -> List[Dict[str, datetime]]:
        """
        Get available time slots across calendars.

        Args:
            calendar_ids: List of calendar IDs to check
            start_time: Start of availability window
            end_time: End of availability window
            slot_duration_minutes: Minimum slot duration

        Returns:
            List of available time slots with start/end times
        """
        # Query free/busy information
        busy_periods = await self.get_free_busy(calendar_ids, start_time, end_time)

        # Merge all busy periods
        all_busy = []
        for cal_busy in busy_periods.values():
            all_busy.extend(cal_busy)

        # Sort by start time
        all_busy.sort(key=lambda x: x["start"])

        # Merge overlapping busy periods
        merged_busy = []
        for busy in all_busy:
            if merged_busy and busy["start"] <= merged_busy[-1]["end"]:
                merged_busy[-1]["end"] = max(merged_busy[-1]["end"], busy["end"])
            else:
                merged_busy.append(busy.copy())

        # Find available slots
        available_slots = []
        slot_duration = timedelta(minutes=slot_duration_minutes)
        current_time = start_time

        for busy in merged_busy:
            # Check if there's a gap before this busy period
            if current_time + slot_duration <= busy["start"]:
                available_slots.append({
                    "start": current_time,
                    "end": busy["start"],
                })
            current_time = max(current_time, busy["end"])

        # Check for availability after last busy period
        if current_time + slot_duration <= end_time:
            available_slots.append({
                "start": current_time,
                "end": end_time,
            })

        return available_slots

    async def get_free_busy(
        self,
        calendar_ids: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, List[Dict[str, datetime]]]:
        """
        Query free/busy information for calendars.

        Args:
            calendar_ids: List of calendar IDs to check
            start_time: Start of query window
            end_time: End of query window

        Returns:
            Dictionary mapping calendar IDs to busy periods
        """
        data = {
            "timeMin": start_time.isoformat() + "Z",
            "timeMax": end_time.isoformat() + "Z",
            "items": [{"id": cal_id} for cal_id in calendar_ids],
        }

        response = await self._request("POST", "freeBusy", json_data=data)

        result = {}
        calendars = response.get("calendars", {})

        for cal_id in calendar_ids:
            cal_data = calendars.get(cal_id, {})
            busy_periods = []

            for busy in cal_data.get("busy", []):
                busy_periods.append({
                    "start": datetime.fromisoformat(busy["start"].replace("Z", "+00:00")),
                    "end": datetime.fromisoformat(busy["end"].replace("Z", "+00:00")),
                })

            result[cal_id] = busy_periods

        return result

    async def check_conflicts(
        self,
        calendar_id: str,
        start_time: datetime,
        end_time: datetime,
        exclude_event_id: Optional[str] = None,
    ) -> List[CalendarEvent]:
        """
        Check for conflicting events in a time range.

        Args:
            calendar_id: Calendar to check
            start_time: Start of time range
            end_time: End of time range
            exclude_event_id: Event ID to exclude from conflicts

        Returns:
            List of conflicting events
        """
        events = await self.list_events(
            calendar_id=calendar_id,
            start_time=start_time,
            end_time=end_time,
        )

        conflicts = []
        for event in events:
            # Skip the excluded event
            if exclude_event_id and event.id == exclude_event_id:
                continue

            # Check for overlap
            if event.start_time < end_time and event.end_time > start_time:
                conflicts.append(event)

        return conflicts

    # =========================================================================
    # Scheduling Operations
    # =========================================================================

    async def schedule_meeting(
        self,
        title: str,
        start_time: datetime,
        end_time: datetime,
        attendee_emails: List[str],
        calendar_id: Optional[str] = None,
        description: str = "",
        location: str = "",
        add_meet_link: bool = True,
        send_invites: bool = True,
    ) -> CalendarEvent:
        """
        Schedule a meeting with attendees.

        Args:
            title: Meeting title
            start_time: Meeting start time
            end_time: Meeting end time
            attendee_emails: List of attendee email addresses
            calendar_id: Calendar to create event in
            description: Meeting description
            location: Meeting location
            add_meet_link: Whether to add Google Meet link
            send_invites: Whether to send calendar invites

        Returns:
            Created CalendarEvent with meeting details
        """
        cal_id = calendar_id or self._default_calendar_id

        event = CalendarEvent(
            id="",
            calendar_id=cal_id,
            title=title,
            description=description,
            location=location,
            start_time=start_time,
            end_time=end_time,
            time_zone=self._time_zone,
            attendees=[{"email": email, "status": "needsAction"} for email in attendee_emails],
            reminders=[{"method": "popup", "minutes": self._default_reminder_minutes}],
        )

        data = self._build_event_data(event)

        # Add Google Meet
        if add_meet_link:
            data["conferenceData"] = {
                "createRequest": {
                    "requestId": hashlib.md5(f"{title}{start_time}".encode()).hexdigest()[:8],
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }

        params = {"sendUpdates": "all" if send_invites else "none"}
        if add_meet_link:
            params["conferenceDataVersion"] = 1

        response = await self._request(
            "POST",
            f"calendars/{cal_id}/events",
            params=params,
            json_data=data,
        )

        return self._convert_event(response, cal_id)

    async def find_meeting_time(
        self,
        attendee_emails: List[str],
        duration_minutes: int,
        start_date: datetime,
        end_date: datetime,
        working_hours_start: int = 9,
        working_hours_end: int = 17,
        preferred_times: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find available meeting times for all attendees.

        Args:
            attendee_emails: Email addresses of attendees
            duration_minutes: Required meeting duration
            start_date: Start of search window
            end_date: End of search window
            working_hours_start: Start of working hours (hour of day)
            working_hours_end: End of working hours (hour of day)
            preferred_times: List of preferred times ("morning", "afternoon", "evening")

        Returns:
            List of suggested meeting times with scores
        """
        # Get free/busy for all attendees
        busy_data = await self.get_free_busy(
            attendee_emails,
            start_date,
            end_date,
        )

        # Find common available slots
        available = await self.get_availability(
            attendee_emails,
            start_date,
            end_date,
            slot_duration_minutes=duration_minutes,
        )

        # Filter by working hours and score suggestions
        suggestions = []
        duration = timedelta(minutes=duration_minutes)

        for slot in available:
            current = slot["start"]
            while current + duration <= slot["end"]:
                hour = current.hour

                # Skip non-working hours
                if hour < working_hours_start or hour >= working_hours_end:
                    current += timedelta(minutes=30)
                    continue

                # Score the time slot
                score = 100

                # Prefer morning/afternoon based on preferences
                if preferred_times:
                    if "morning" in preferred_times and 9 <= hour < 12:
                        score += 10
                    if "afternoon" in preferred_times and 12 <= hour < 17:
                        score += 10

                # Slight preference for common meeting times
                if hour in [9, 10, 14, 15]:
                    score += 5

                # Avoid lunch hours
                if 12 <= hour < 13:
                    score -= 10

                suggestions.append({
                    "start": current,
                    "end": current + duration,
                    "score": score,
                    "attendee_count": len(attendee_emails),
                    "all_available": True,
                })

                current += timedelta(minutes=30)

        # Sort by score descending
        suggestions.sort(key=lambda x: x["score"], reverse=True)

        return suggestions[:10]  # Return top 10 suggestions

    # =========================================================================
    # Sync Operations
    # =========================================================================

    async def sync_events(
        self,
        calendar_id: Optional[str] = None,
        full_sync: bool = False,
    ) -> Tuple[List[CalendarEvent], List[str], Optional[str]]:
        """
        Perform incremental sync of calendar events.

        Args:
            calendar_id: Calendar to sync
            full_sync: Force full sync instead of incremental

        Returns:
            Tuple of (updated events, deleted event IDs, next sync token)
        """
        cal_id = calendar_id or self._default_calendar_id
        updated_events = []
        deleted_ids = []

        params = {
            "maxResults": self.DEFAULT_MAX_RESULTS,
            "singleEvents": False,  # Include recurring events
            "showDeleted": True,  # Include deleted events
        }

        # Use sync token for incremental sync
        sync_token = None if full_sync else self._sync_tokens.get(cal_id)

        if sync_token:
            params["syncToken"] = sync_token
        else:
            # Full sync - get events from past month
            params["timeMin"] = (datetime.utcnow() - timedelta(days=30)).isoformat() + "Z"

        page_token = None

        while True:
            if page_token:
                params["pageToken"] = page_token

            try:
                response = await self._request("GET", f"calendars/{cal_id}/events", params)
            except IntegrationError as e:
                if "Sync token is no longer valid" in str(e):
                    # Token expired, do full sync
                    logger.info(f"Sync token expired for calendar {cal_id}, performing full sync")
                    return await self.sync_events(cal_id, full_sync=True)
                raise

            for item in response.get("items", []):
                if item.get("status") == "cancelled":
                    deleted_ids.append(item.get("id"))
                else:
                    event = self._convert_event(item, cal_id)
                    if event:
                        updated_events.append(event)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        # Store new sync token
        new_sync_token = response.get("nextSyncToken")
        if new_sync_token:
            self._sync_tokens[cal_id] = new_sync_token

        return updated_events, deleted_ids, new_sync_token

    async def get_sync_state(self, resource_type: str) -> Optional[SyncState]:
        """
        Get current sync state for a resource type.

        Args:
            resource_type: Type of resource (e.g., "events")

        Returns:
            SyncState object or None
        """
        if resource_type == "events":
            cal_id = self._default_calendar_id
            sync_token = self._sync_tokens.get(cal_id)

            if sync_token:
                return SyncState(
                    resource_type=resource_type,
                    cursor=sync_token,
                    last_sync=datetime.utcnow(),
                    full_sync_needed=False,
                )

        return None

    # =========================================================================
    # Watch/Notification Operations
    # =========================================================================

    async def setup_watch(
        self,
        calendar_id: str,
        webhook_url: str,
        channel_id: str,
        ttl_seconds: int = 604800,  # 7 days
    ) -> Dict[str, Any]:
        """
        Set up push notifications for calendar changes.

        Args:
            calendar_id: Calendar to watch
            webhook_url: URL to receive notifications
            channel_id: Unique channel identifier
            ttl_seconds: Time-to-live for the watch

        Returns:
            Watch subscription details
        """
        data = {
            "id": channel_id,
            "type": "web_hook",
            "address": webhook_url,
            "params": {
                "ttl": str(ttl_seconds),
            },
        }

        response = await self._request(
            "POST",
            f"calendars/{calendar_id}/events/watch",
            json_data=data,
        )

        return {
            "channel_id": response.get("id"),
            "resource_id": response.get("resourceId"),
            "resource_uri": response.get("resourceUri"),
            "expiration": datetime.fromtimestamp(int(response.get("expiration", 0)) / 1000),
        }

    async def stop_watch(self, channel_id: str, resource_id: str) -> bool:
        """
        Stop push notifications for a channel.

        Args:
            channel_id: Channel identifier
            resource_id: Resource identifier from watch setup

        Returns:
            True if successful
        """
        data = {
            "id": channel_id,
            "resourceId": resource_id,
        }

        await self._request("POST", "channels/stop", json_data=data)
        return True

    # =========================================================================
    # OAuth2 Helper Methods
    # =========================================================================

    @classmethod
    def get_authorization_url(
        cls,
        client_id: str,
        redirect_uri: str,
        state: str,
        scopes: Optional[List[str]] = None,
    ) -> str:
        """
        Generate OAuth2 authorization URL.

        Args:
            client_id: OAuth2 client ID
            redirect_uri: Redirect URI after authorization
            state: State parameter for security
            scopes: OAuth2 scopes to request

        Returns:
            Authorization URL
        """
        default_scopes = [
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/calendar.events",
        ]

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes or default_scopes),
            "state": state,
            "access_type": "offline",
            "prompt": "consent",  # Always show consent screen for refresh token
        }

        return f"{cls.AUTH_URL}?{urlencode(params)}"

    @classmethod
    async def exchange_code(
        cls,
        code: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            redirect_uri: Redirect URI used in authorization

        Returns:
            Token response with access_token, refresh_token, etc.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                cls.TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                },
            ) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise AuthenticationError(
                        f"Token exchange failed: {error_data.get('error_description', 'Unknown error')}"
                    )

                data = await response.json()
                return {
                    "access_token": data["access_token"],
                    "refresh_token": data.get("refresh_token"),
                    "token_expiry": (
                        datetime.utcnow() + timedelta(seconds=data.get("expires_in", 3600))
                    ).isoformat(),
                    "token_type": data.get("token_type", "Bearer"),
                    "scope": data.get("scope"),
                }

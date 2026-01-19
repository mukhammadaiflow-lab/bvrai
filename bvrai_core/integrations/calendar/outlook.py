"""
Microsoft Outlook Calendar Integration

This module provides Microsoft Outlook/Office 365 Calendar integration for the
voice agent platform, enabling calendar synchronization, availability checking,
and meeting scheduling through Microsoft Graph API.

Features:
- OAuth2 authentication with Azure AD
- Calendar listing and event management
- Free/busy availability queries via scheduling API
- Teams meeting integration
- Recurring event support with patterns
- Delta sync for efficient updates
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


class OutlookCalendarProvider(CalendarProvider):
    """
    Microsoft Outlook/Office 365 Calendar integration provider.

    Implements Microsoft Graph Calendar API functionality including:
    - Calendar and event CRUD operations
    - Scheduling API for availability queries
    - Teams meeting creation
    - Delta queries for incremental sync
    - Change notifications with subscriptions

    OAuth2 Scopes Required:
    - Calendars.ReadWrite (full calendar access)
    - Calendars.Read (read-only access)
    - MailboxSettings.Read (for time zone info)
    - OnlineMeetings.ReadWrite (for Teams meetings)
    """

    PROVIDER_NAME = "outlook_calendar"
    AUTH_TYPE = AuthType.OAUTH2

    # Microsoft Graph API endpoints
    BASE_URL = "https://graph.microsoft.com/v1.0"
    TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    AUTH_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"

    # API configuration
    DEFAULT_PAGE_SIZE = 50
    MAX_PAGE_SIZE = 999
    MAX_ATTENDEES = 500
    RATE_LIMIT_DELAY = 0.5
    MAX_RETRIES = 3

    # Recurrence patterns
    RECURRENCE_TYPES = {
        "daily": "daily",
        "weekly": "weekly",
        "absoluteMonthly": "monthly",
        "relativeMonthly": "monthly",
        "absoluteYearly": "yearly",
        "relativeYearly": "yearly",
    }

    # Days of week mapping
    DAYS_OF_WEEK = {
        0: "monday",
        1: "tuesday",
        2: "wednesday",
        3: "thursday",
        4: "friday",
        5: "saturday",
        6: "sunday",
    }

    def __init__(
        self,
        integration_id: str,
        organization_id: str,
        credentials: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Outlook Calendar provider.

        Args:
            integration_id: Unique integration identifier
            organization_id: Organization this integration belongs to
            credentials: OAuth2 credentials including:
                - access_token: Current access token
                - refresh_token: Token for refreshing access
                - client_id: Azure AD application ID
                - client_secret: Azure AD client secret
                - tenant_id: Azure AD tenant ID (or "common")
                - token_expiry: Token expiration timestamp
            settings: Optional settings including:
                - default_calendar_id: Default calendar for operations
                - time_zone: Default time zone
                - default_reminder_minutes: Default reminder time
                - auto_add_teams: Automatically add Teams meetings
        """
        super().__init__(integration_id, organization_id, credentials, settings or {})

        self._access_token = credentials.get("access_token")
        self._refresh_token = credentials.get("refresh_token")
        self._client_id = credentials.get("client_id")
        self._client_secret = credentials.get("client_secret")
        self._tenant_id = credentials.get("tenant_id", "common")
        self._token_expiry = credentials.get("token_expiry")

        # Settings
        self._default_calendar_id = self.settings.get("default_calendar_id")
        self._time_zone = self.settings.get("time_zone", "UTC")
        self._default_reminder_minutes = self.settings.get("default_reminder_minutes", 15)
        self._auto_add_teams = self.settings.get("auto_add_teams", False)

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        # Delta links for incremental sync
        self._delta_links: Dict[str, str] = {}

        # Rate limiting
        self._last_request_time: Optional[datetime] = None

        # User info cache
        self._user_id: Optional[str] = None
        self._user_email: Optional[str] = None

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
        token_url = self.TOKEN_URL.format(tenant=self._tenant_id)

        try:
            async with session.post(
                token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": "https://graph.microsoft.com/.default offline_access",
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

                # Update refresh token if new one provided
                if data.get("refresh_token"):
                    self._refresh_token = data["refresh_token"]

                # Update credentials
                self.credentials["access_token"] = self._access_token
                self.credentials["token_expiry"] = self._token_expiry.isoformat()
                if data.get("refresh_token"):
                    self.credentials["refresh_token"] = self._refresh_token

                # Emit refresh event
                if self._on_token_refresh:
                    await self._on_token_refresh(
                        self.integration_id,
                        {
                            "access_token": self._access_token,
                            "refresh_token": self._refresh_token,
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
        headers: Optional[Dict[str, str]] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to the Microsoft Graph API.

        Args:
            method: HTTP method
            endpoint: API endpoint path (without base URL)
            params: Query parameters
            json_data: JSON request body
            headers: Additional headers
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

        # Handle full URLs (for delta links and @odata.nextLink)
        if endpoint.startswith("http"):
            url = endpoint
        else:
            url = f"{self.BASE_URL}/{endpoint}"

        request_headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "Prefer": f'outlook.timezone="{self._time_zone}"',
        }
        if headers:
            request_headers.update(headers)

        try:
            self._last_request_time = datetime.utcnow()

            async with session.request(
                method,
                url,
                headers=request_headers,
                params=params,
                json=json_data,
            ) as response:
                # Handle rate limiting (429 or 503)
                if response.status in [429, 503]:
                    if retry_count < self.MAX_RETRIES:
                        retry_after = int(response.headers.get("Retry-After", "5"))
                        logger.warning(
                            f"Rate limited, retrying after {retry_after}s "
                            f"(attempt {retry_count + 1}/{self.MAX_RETRIES})"
                        )
                        await asyncio.sleep(retry_after)
                        return await self._request(
                            method, endpoint, params, json_data, headers, retry_count + 1
                        )
                    raise RateLimitError("Microsoft Graph API rate limit exceeded")

                # Handle authentication errors
                if response.status == 401:
                    if retry_count < 1:
                        await self._refresh_access_token()
                        return await self._request(
                            method, endpoint, params, json_data, headers, retry_count + 1
                        )
                    raise AuthenticationError("Microsoft Graph authentication failed")

                # Handle not found
                if response.status == 404:
                    return {}

                # Handle other errors
                if response.status >= 400:
                    try:
                        error_data = await response.json()
                        error_info = error_data.get("error", {})
                        error_message = error_info.get("message", "Unknown error")
                        error_code = error_info.get("code", "")
                    except Exception:
                        error_message = await response.text()
                        error_code = ""

                    raise IntegrationError(
                        f"Microsoft Graph API error ({response.status}): "
                        f"{error_code} - {error_message}"
                    )

                # Return empty dict for 204 No Content
                if response.status == 204:
                    return {}

                return await response.json()

        except aiohttp.ClientError as e:
            if retry_count < self.MAX_RETRIES:
                await asyncio.sleep(2 ** retry_count)
                return await self._request(
                    method, endpoint, params, json_data, headers, retry_count + 1
                )
            raise IntegrationError(f"Network error: {e}")

    async def _get_user_info(self) -> Tuple[str, str]:
        """Get current user ID and email."""
        if not self._user_id or not self._user_email:
            response = await self._request("GET", "me", {"$select": "id,mail,userPrincipalName"})
            self._user_id = response.get("id")
            self._user_email = response.get("mail") or response.get("userPrincipalName")
        return self._user_id, self._user_email

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
            # Verify by fetching user profile
            await self._get_user_info()
            self._status = IntegrationStatus.CONNECTED
            logger.info(f"Connected to Outlook Calendar for integration {self.integration_id}")
            return True

        except AuthenticationError:
            self._status = IntegrationStatus.AUTH_FAILED
            return False
        except Exception as e:
            self._status = IntegrationStatus.ERROR
            logger.error(f"Failed to connect to Outlook Calendar: {e}")
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
            self._delta_links.clear()
            self._user_id = None
            self._user_email = None
            logger.info(f"Disconnected from Outlook Calendar for integration {self.integration_id}")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from Outlook Calendar: {e}")
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
            start = datetime.utcnow()
            await self._get_user_info()
            latency_ms = int((datetime.utcnow() - start).total_seconds() * 1000)

            health["status"] = "healthy"
            health["details"] = {
                "api_latency_ms": latency_ms,
                "user_id": self._user_id,
                "user_email": self._user_email,
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
        endpoint = "me/calendars"

        while endpoint:
            response = await self._request(
                "GET",
                endpoint,
                {"$top": self.DEFAULT_PAGE_SIZE},
            )

            for item in response.get("value", []):
                calendars.append(self._convert_calendar(item))

            endpoint = response.get("@odata.nextLink")

        return calendars

    async def get_calendar(self, calendar_id: str) -> Optional[Calendar]:
        """
        Get a specific calendar by ID.

        Args:
            calendar_id: Calendar identifier

        Returns:
            Calendar object or None if not found
        """
        response = await self._request("GET", f"me/calendars/{calendar_id}")
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
            description: Calendar description (not supported by Graph API)
            time_zone: Calendar time zone

        Returns:
            Created Calendar object
        """
        data = {"name": name}

        response = await self._request("POST", "me/calendars", json_data=data)
        return self._convert_calendar(response)

    async def delete_calendar(self, calendar_id: str) -> bool:
        """
        Delete a calendar.

        Args:
            calendar_id: Calendar identifier

        Returns:
            True if deletion successful
        """
        await self._request("DELETE", f"me/calendars/{calendar_id}")
        return True

    def _convert_calendar(self, data: Dict[str, Any]) -> Calendar:
        """Convert Microsoft Graph API response to Calendar object."""
        # Determine access role from permissions
        access_role = "reader"
        if data.get("canEdit"):
            access_role = "writer"
        if data.get("isDefaultCalendar"):
            access_role = "owner"

        return Calendar(
            id=data["id"],
            name=data.get("name", ""),
            description="",  # Graph API doesn't support calendar description
            time_zone=self._time_zone,  # Calendars don't have individual time zones
            color=data.get("hexColor") or data.get("color"),
            is_primary=data.get("isDefaultCalendar", False),
            access_role=access_role,
            metadata={
                "canShare": data.get("canShare"),
                "canViewPrivateItems": data.get("canViewPrivateItems"),
                "canEdit": data.get("canEdit"),
                "changeKey": data.get("changeKey"),
                "owner": data.get("owner"),
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
            cursor: Pagination cursor (skip token)

        Returns:
            List of CalendarEvent objects
        """
        cal_id = calendar_id or self._default_calendar_id
        events = []

        # Build endpoint
        if cal_id:
            endpoint = f"me/calendars/{cal_id}/events"
        else:
            endpoint = "me/events"

        # Build query parameters
        params = {
            "$top": min(limit, self.MAX_PAGE_SIZE),
            "$orderby": "start/dateTime",
            "$select": "id,subject,body,bodyPreview,start,end,location,locations,"
                       "attendees,organizer,isAllDay,isCancelled,isOnlineMeeting,"
                       "onlineMeeting,onlineMeetingUrl,recurrence,showAs,importance,"
                       "sensitivity,categories,createdDateTime,lastModifiedDateTime",
        }

        # Add time filter
        filters = []
        if start_time:
            filters.append(f"start/dateTime ge '{start_time.isoformat()}'")
        if end_time:
            filters.append(f"end/dateTime le '{end_time.isoformat()}'")

        if filters:
            params["$filter"] = " and ".join(filters)

        if cursor:
            params["$skip"] = cursor

        response = await self._request("GET", endpoint, params)

        for item in response.get("value", []):
            event = self._convert_event(item, cal_id or "primary")
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

        if cal_id:
            endpoint = f"me/calendars/{cal_id}/events/{event_id}"
        else:
            endpoint = f"me/events/{event_id}"

        response = await self._request("GET", endpoint)

        if not response:
            return None

        return self._convert_event(response, cal_id or "primary")

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

        # Add Teams meeting if configured
        if self._auto_add_teams and not event.meeting_link:
            data["isOnlineMeeting"] = True
            data["onlineMeetingProvider"] = "teamsForBusiness"

        if cal_id:
            endpoint = f"me/calendars/{cal_id}/events"
        else:
            endpoint = "me/events"

        response = await self._request("POST", endpoint, json_data=data)
        return self._convert_event(response, cal_id or "primary")

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

        if cal_id:
            endpoint = f"me/calendars/{cal_id}/events/{event.id}"
        else:
            endpoint = f"me/events/{event.id}"

        response = await self._request("PATCH", endpoint, json_data=data)
        return self._convert_event(response, cal_id or "primary")

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
            send_updates: Notification setting (ignored for Graph API)

        Returns:
            True if deletion successful
        """
        cal_id = calendar_id or self._default_calendar_id

        if cal_id:
            endpoint = f"me/calendars/{cal_id}/events/{event_id}"
        else:
            endpoint = f"me/events/{event_id}"

        await self._request("DELETE", endpoint)
        return True

    def _build_event_data(self, event: CalendarEvent) -> Dict[str, Any]:
        """Build Microsoft Graph API event data from CalendarEvent."""
        data = {
            "subject": event.title,
            "body": {
                "contentType": "text",
                "content": event.description or "",
            },
        }

        # Handle all-day vs timed events
        if event.all_day:
            data["isAllDay"] = True
            data["start"] = {
                "dateTime": event.start_time.strftime("%Y-%m-%dT00:00:00"),
                "timeZone": event.time_zone or self._time_zone,
            }
            data["end"] = {
                "dateTime": event.end_time.strftime("%Y-%m-%dT00:00:00"),
                "timeZone": event.time_zone or self._time_zone,
            }
        else:
            data["start"] = {
                "dateTime": event.start_time.isoformat(),
                "timeZone": event.time_zone or self._time_zone,
            }
            data["end"] = {
                "dateTime": event.end_time.isoformat(),
                "timeZone": event.time_zone or self._time_zone,
            }

        # Add location
        if event.location:
            data["location"] = {"displayName": event.location}

        # Add attendees
        if event.attendees:
            data["attendees"] = []
            for attendee in event.attendees[:self.MAX_ATTENDEES]:
                attendee_type = "required"
                if attendee.get("optional"):
                    attendee_type = "optional"

                data["attendees"].append({
                    "emailAddress": {
                        "address": attendee.get("email"),
                        "name": attendee.get("name", ""),
                    },
                    "type": attendee_type,
                })

        # Add reminder
        if event.reminders:
            # Graph API supports single reminder
            reminder = event.reminders[0] if event.reminders else None
            if reminder:
                data["isReminderOn"] = True
                data["reminderMinutesBeforeStart"] = reminder.get(
                    "minutes", self._default_reminder_minutes
                )
        else:
            data["isReminderOn"] = True
            data["reminderMinutesBeforeStart"] = self._default_reminder_minutes

        # Add recurrence
        if event.recurrence:
            data["recurrence"] = self._build_recurrence(event.recurrence)

        # Add sensitivity
        if event.visibility:
            sensitivity_map = {
                "public": "normal",
                "private": "private",
                "confidential": "confidential",
            }
            data["sensitivity"] = sensitivity_map.get(event.visibility, "normal")

        # Add show as (free/busy status)
        if event.status:
            show_as_map = {
                "free": "free",
                "busy": "busy",
                "tentative": "tentative",
                "oof": "oof",
            }
            data["showAs"] = show_as_map.get(event.status, "busy")

        return data

    def _build_recurrence(self, recurrence_rules: List[str]) -> Dict[str, Any]:
        """Build Graph API recurrence pattern from RRULE strings."""
        # Parse simple RRULE format
        # This is a simplified implementation
        recurrence = {
            "pattern": {},
            "range": {"type": "noEnd"},
        }

        for rule in recurrence_rules:
            if not rule.startswith("RRULE:"):
                continue

            rule_parts = rule[6:].split(";")
            params = {}
            for part in rule_parts:
                key, value = part.split("=")
                params[key] = value

            freq = params.get("FREQ", "").lower()
            if freq == "daily":
                recurrence["pattern"]["type"] = "daily"
                recurrence["pattern"]["interval"] = int(params.get("INTERVAL", 1))

            elif freq == "weekly":
                recurrence["pattern"]["type"] = "weekly"
                recurrence["pattern"]["interval"] = int(params.get("INTERVAL", 1))
                if "BYDAY" in params:
                    day_map = {
                        "MO": "monday", "TU": "tuesday", "WE": "wednesday",
                        "TH": "thursday", "FR": "friday", "SA": "saturday", "SU": "sunday"
                    }
                    days = params["BYDAY"].split(",")
                    recurrence["pattern"]["daysOfWeek"] = [
                        day_map.get(d, "monday") for d in days
                    ]

            elif freq == "monthly":
                recurrence["pattern"]["type"] = "absoluteMonthly"
                recurrence["pattern"]["interval"] = int(params.get("INTERVAL", 1))
                if "BYMONTHDAY" in params:
                    recurrence["pattern"]["dayOfMonth"] = int(params["BYMONTHDAY"])

            elif freq == "yearly":
                recurrence["pattern"]["type"] = "absoluteYearly"
                recurrence["pattern"]["interval"] = int(params.get("INTERVAL", 1))

            # Handle count/until
            if "COUNT" in params:
                recurrence["range"]["type"] = "numbered"
                recurrence["range"]["numberOfOccurrences"] = int(params["COUNT"])
            elif "UNTIL" in params:
                recurrence["range"]["type"] = "endDate"
                # Parse UNTIL date
                until = params["UNTIL"]
                if len(until) == 8:  # YYYYMMDD format
                    recurrence["range"]["endDate"] = f"{until[:4]}-{until[4:6]}-{until[6:8]}"

        return recurrence

    def _convert_event(
        self,
        data: Dict[str, Any],
        calendar_id: str,
    ) -> Optional[CalendarEvent]:
        """Convert Microsoft Graph API response to CalendarEvent object."""
        if not data:
            return None

        # Skip cancelled events
        if data.get("isCancelled"):
            return None

        # Parse start/end times
        start_data = data.get("start", {})
        end_data = data.get("end", {})

        all_day = data.get("isAllDay", False)

        try:
            start_str = start_data.get("dateTime", "")
            end_str = end_data.get("dateTime", "")

            # Graph API returns ISO format without timezone suffix
            if "Z" not in start_str and "+" not in start_str:
                start_time = datetime.fromisoformat(start_str)
                end_time = datetime.fromisoformat(end_str)
            else:
                start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                end_time = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        except ValueError as e:
            logger.warning(f"Could not parse event times: {e}")
            return None

        # Parse attendees
        attendees = []
        for attendee in data.get("attendees", []):
            email_addr = attendee.get("emailAddress", {})
            status_data = attendee.get("status", {})

            attendees.append({
                "email": email_addr.get("address"),
                "name": email_addr.get("name"),
                "status": status_data.get("response", "none"),
                "type": attendee.get("type", "required"),
                "optional": attendee.get("type") == "optional",
            })

        # Parse organizer
        organizer = None
        organizer_data = data.get("organizer")
        if organizer_data:
            email_addr = organizer_data.get("emailAddress", {})
            organizer = {
                "email": email_addr.get("address"),
                "name": email_addr.get("name"),
            }

        # Parse reminders
        reminders = []
        if data.get("isReminderOn"):
            reminders.append({
                "method": "popup",
                "minutes": data.get("reminderMinutesBeforeStart", 15),
            })

        # Get meeting link
        meeting_link = None
        if data.get("isOnlineMeeting"):
            meeting_link = data.get("onlineMeetingUrl")
            if not meeting_link:
                online_meeting = data.get("onlineMeeting", {})
                meeting_link = online_meeting.get("joinUrl")

        # Parse recurrence
        recurrence = None
        recurrence_data = data.get("recurrence")
        if recurrence_data:
            recurrence = self._parse_recurrence(recurrence_data)

        # Parse visibility
        visibility = None
        sensitivity = data.get("sensitivity")
        if sensitivity:
            visibility_map = {
                "normal": "public",
                "personal": "private",
                "private": "private",
                "confidential": "confidential",
            }
            visibility = visibility_map.get(sensitivity, "public")

        # Parse status
        status = data.get("showAs")

        # Parse location
        location = None
        location_data = data.get("location")
        if location_data:
            location = location_data.get("displayName")

        return CalendarEvent(
            id=data.get("id", ""),
            calendar_id=calendar_id,
            title=data.get("subject", "(No title)"),
            description=data.get("body", {}).get("content"),
            location=location,
            start_time=start_time,
            end_time=end_time,
            time_zone=start_data.get("timeZone"),
            all_day=all_day,
            attendees=attendees,
            organizer=organizer,
            reminders=reminders,
            recurrence=recurrence,
            meeting_link=meeting_link,
            status=status,
            visibility=visibility,
            created_at=datetime.fromisoformat(
                data.get("createdDateTime", "").replace("Z", "+00:00")
            ) if data.get("createdDateTime") else None,
            updated_at=datetime.fromisoformat(
                data.get("lastModifiedDateTime", "").replace("Z", "+00:00")
            ) if data.get("lastModifiedDateTime") else None,
            metadata={
                "importance": data.get("importance"),
                "categories": data.get("categories", []),
                "bodyPreview": data.get("bodyPreview"),
                "changeKey": data.get("changeKey"),
                "webLink": data.get("webLink"),
            },
        )

    def _parse_recurrence(self, data: Dict[str, Any]) -> List[str]:
        """Convert Graph API recurrence to RRULE format."""
        rules = []
        pattern = data.get("pattern", {})
        range_data = data.get("range", {})

        pattern_type = pattern.get("type", "")
        interval = pattern.get("interval", 1)

        # Build RRULE
        parts = []

        if pattern_type == "daily":
            parts.append("FREQ=DAILY")
        elif pattern_type == "weekly":
            parts.append("FREQ=WEEKLY")
            days = pattern.get("daysOfWeek", [])
            if days:
                day_map = {
                    "monday": "MO", "tuesday": "TU", "wednesday": "WE",
                    "thursday": "TH", "friday": "FR", "saturday": "SA", "sunday": "SU"
                }
                byday = ",".join([day_map.get(d, "MO") for d in days])
                parts.append(f"BYDAY={byday}")
        elif pattern_type in ["absoluteMonthly", "relativeMonthly"]:
            parts.append("FREQ=MONTHLY")
            if pattern.get("dayOfMonth"):
                parts.append(f"BYMONTHDAY={pattern['dayOfMonth']}")
        elif pattern_type in ["absoluteYearly", "relativeYearly"]:
            parts.append("FREQ=YEARLY")

        if interval > 1:
            parts.append(f"INTERVAL={interval}")

        # Handle range
        range_type = range_data.get("type", "")
        if range_type == "numbered":
            parts.append(f"COUNT={range_data.get('numberOfOccurrences', 1)}")
        elif range_type == "endDate":
            end_date = range_data.get("endDate", "")
            if end_date:
                until = end_date.replace("-", "")
                parts.append(f"UNTIL={until}")

        if parts:
            rules.append("RRULE:" + ";".join(parts))

        return rules

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
            calendar_ids: List of calendar IDs or email addresses to check
            start_time: Start of availability window
            end_time: End of availability window
            slot_duration_minutes: Minimum slot duration

        Returns:
            List of available time slots with start/end times
        """
        # Query free/busy information using scheduling API
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
            if current_time + slot_duration <= busy["start"]:
                available_slots.append({
                    "start": current_time,
                    "end": busy["start"],
                })
            current_time = max(current_time, busy["end"])

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
        Query free/busy information using scheduling API.

        Args:
            calendar_ids: List of email addresses or calendar IDs to check
            start_time: Start of query window
            end_time: End of query window

        Returns:
            Dictionary mapping identifiers to busy periods
        """
        # Use getSchedule endpoint
        data = {
            "schedules": calendar_ids,
            "startTime": {
                "dateTime": start_time.isoformat(),
                "timeZone": self._time_zone,
            },
            "endTime": {
                "dateTime": end_time.isoformat(),
                "timeZone": self._time_zone,
            },
            "availabilityViewInterval": 30,  # 30-minute intervals
        }

        response = await self._request("POST", "me/calendar/getSchedule", json_data=data)

        result = {}
        for schedule in response.get("value", []):
            schedule_id = schedule.get("scheduleId", "")
            busy_periods = []

            for item in schedule.get("scheduleItems", []):
                if item.get("status") in ["busy", "tentative", "oof"]:
                    start = item.get("start", {})
                    end = item.get("end", {})

                    busy_periods.append({
                        "start": datetime.fromisoformat(
                            start.get("dateTime", "").replace("Z", "+00:00")
                        ),
                        "end": datetime.fromisoformat(
                            end.get("dateTime", "").replace("Z", "+00:00")
                        ),
                        "status": item.get("status"),
                    })

            result[schedule_id] = busy_periods

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
            if exclude_event_id and event.id == exclude_event_id:
                continue

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
        add_teams_link: bool = True,
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
            add_teams_link: Whether to add Teams meeting link
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
            attendees=[{"email": email, "status": "none"} for email in attendee_emails],
            reminders=[{"method": "popup", "minutes": self._default_reminder_minutes}],
        )

        data = self._build_event_data(event)

        # Add Teams meeting
        if add_teams_link:
            data["isOnlineMeeting"] = True
            data["onlineMeetingProvider"] = "teamsForBusiness"

        if cal_id:
            endpoint = f"me/calendars/{cal_id}/events"
        else:
            endpoint = "me/events"

        # Note: Graph API always sends invites to attendees
        response = await self._request("POST", endpoint, json_data=data)
        return self._convert_event(response, cal_id or "primary")

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
        Find available meeting times for all attendees using findMeetingTimes API.

        Args:
            attendee_emails: Email addresses of attendees
            duration_minutes: Required meeting duration
            start_date: Start of search window
            end_date: End of search window
            working_hours_start: Start of working hours
            working_hours_end: End of working hours
            preferred_times: List of preferred times

        Returns:
            List of suggested meeting times with confidence scores
        """
        # Build attendees list
        attendees = [
            {
                "emailAddress": {"address": email},
                "type": "required",
            }
            for email in attendee_emails
        ]

        # Build time constraint
        data = {
            "attendees": attendees,
            "timeConstraint": {
                "activityDomain": "work",
                "timeSlots": [
                    {
                        "start": {
                            "dateTime": start_date.isoformat(),
                            "timeZone": self._time_zone,
                        },
                        "end": {
                            "dateTime": end_date.isoformat(),
                            "timeZone": self._time_zone,
                        },
                    }
                ],
            },
            "meetingDuration": f"PT{duration_minutes}M",
            "returnSuggestionReasons": True,
            "minimumAttendeePercentage": 100,
        }

        response = await self._request("POST", "me/findMeetingTimes", json_data=data)

        suggestions = []
        for suggestion in response.get("meetingTimeSuggestions", []):
            meeting_time = suggestion.get("meetingTimeSlot", {})
            start_data = meeting_time.get("start", {})
            end_data = meeting_time.get("end", {})

            start = datetime.fromisoformat(start_data.get("dateTime", ""))
            end = datetime.fromisoformat(end_data.get("dateTime", ""))

            # Calculate score based on confidence
            confidence = suggestion.get("confidence", 0)
            score = int(confidence * 100)

            suggestions.append({
                "start": start,
                "end": end,
                "score": score,
                "confidence": confidence,
                "attendee_count": len(attendee_emails),
                "organizer_availability": suggestion.get("organizerAvailability"),
                "suggestion_reason": suggestion.get("suggestionReason"),
                "attendee_availability": [
                    {
                        "email": att.get("attendee", {}).get("emailAddress", {}).get("address"),
                        "availability": att.get("availability"),
                    }
                    for att in suggestion.get("attendeeAvailability", [])
                ],
            })

        return suggestions

    # =========================================================================
    # Sync Operations
    # =========================================================================

    async def sync_events(
        self,
        calendar_id: Optional[str] = None,
        full_sync: bool = False,
    ) -> Tuple[List[CalendarEvent], List[str], Optional[str]]:
        """
        Perform incremental sync of calendar events using delta queries.

        Args:
            calendar_id: Calendar to sync
            full_sync: Force full sync instead of incremental

        Returns:
            Tuple of (updated events, deleted event IDs, next delta link)
        """
        cal_id = calendar_id or self._default_calendar_id or "primary"
        updated_events = []
        deleted_ids = []

        # Use delta link for incremental sync
        delta_link = None if full_sync else self._delta_links.get(cal_id)

        if delta_link:
            endpoint = delta_link
            params = None
        else:
            # Full sync
            if cal_id and cal_id != "primary":
                endpoint = f"me/calendars/{cal_id}/events/delta"
            else:
                endpoint = "me/calendarView/delta"

            # For full sync, get events from past month to future month
            start_date = datetime.utcnow() - timedelta(days=30)
            end_date = datetime.utcnow() + timedelta(days=365)

            params = {
                "startDateTime": start_date.isoformat() + "Z",
                "endDateTime": end_date.isoformat() + "Z",
            }

        # Paginate through results
        while endpoint:
            try:
                response = await self._request("GET", endpoint, params)
            except IntegrationError as e:
                if "resyncRequired" in str(e) or "syncStateNotFound" in str(e):
                    logger.info(f"Delta token invalid for calendar {cal_id}, performing full sync")
                    return await self.sync_events(cal_id, full_sync=True)
                raise

            for item in response.get("value", []):
                # Check if deleted
                if item.get("@removed"):
                    deleted_ids.append(item.get("id"))
                else:
                    event = self._convert_event(item, cal_id)
                    if event:
                        updated_events.append(event)

            # Get next page or delta link
            endpoint = response.get("@odata.nextLink")
            params = None  # Params only for initial request

            if not endpoint:
                new_delta_link = response.get("@odata.deltaLink")
                if new_delta_link:
                    self._delta_links[cal_id] = new_delta_link

        return updated_events, deleted_ids, self._delta_links.get(cal_id)

    async def get_sync_state(self, resource_type: str) -> Optional[SyncState]:
        """
        Get current sync state for a resource type.

        Args:
            resource_type: Type of resource (e.g., "events")

        Returns:
            SyncState object or None
        """
        if resource_type == "events":
            cal_id = self._default_calendar_id or "primary"
            delta_link = self._delta_links.get(cal_id)

            if delta_link:
                return SyncState(
                    resource_type=resource_type,
                    cursor=delta_link,
                    last_sync=datetime.utcnow(),
                    full_sync_needed=False,
                )

        return None

    # =========================================================================
    # Subscription/Notification Operations
    # =========================================================================

    async def setup_subscription(
        self,
        calendar_id: str,
        webhook_url: str,
        expiration_minutes: int = 4230,  # Max ~3 days
    ) -> Dict[str, Any]:
        """
        Set up change notifications subscription.

        Args:
            calendar_id: Calendar to subscribe to
            webhook_url: URL to receive notifications
            expiration_minutes: Subscription lifetime in minutes

        Returns:
            Subscription details
        """
        expiration = datetime.utcnow() + timedelta(minutes=expiration_minutes)

        if calendar_id and calendar_id != "primary":
            resource = f"me/calendars/{calendar_id}/events"
        else:
            resource = "me/events"

        data = {
            "changeType": "created,updated,deleted",
            "notificationUrl": webhook_url,
            "resource": resource,
            "expirationDateTime": expiration.isoformat() + "Z",
            "clientState": hashlib.md5(
                f"{self.integration_id}{calendar_id}".encode()
            ).hexdigest()[:32],
        }

        response = await self._request("POST", "subscriptions", json_data=data)

        return {
            "subscription_id": response.get("id"),
            "resource": response.get("resource"),
            "expiration": datetime.fromisoformat(
                response.get("expirationDateTime", "").replace("Z", "+00:00")
            ),
            "client_state": response.get("clientState"),
        }

    async def renew_subscription(
        self,
        subscription_id: str,
        expiration_minutes: int = 4230,
    ) -> Dict[str, Any]:
        """
        Renew an existing subscription.

        Args:
            subscription_id: Subscription to renew
            expiration_minutes: New lifetime in minutes

        Returns:
            Updated subscription details
        """
        expiration = datetime.utcnow() + timedelta(minutes=expiration_minutes)

        data = {
            "expirationDateTime": expiration.isoformat() + "Z",
        }

        response = await self._request(
            "PATCH",
            f"subscriptions/{subscription_id}",
            json_data=data,
        )

        return {
            "subscription_id": response.get("id"),
            "expiration": datetime.fromisoformat(
                response.get("expirationDateTime", "").replace("Z", "+00:00")
            ),
        }

    async def delete_subscription(self, subscription_id: str) -> bool:
        """
        Delete a subscription.

        Args:
            subscription_id: Subscription to delete

        Returns:
            True if successful
        """
        await self._request("DELETE", f"subscriptions/{subscription_id}")
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
        tenant_id: str = "common",
        scopes: Optional[List[str]] = None,
    ) -> str:
        """
        Generate OAuth2 authorization URL.

        Args:
            client_id: Azure AD application ID
            redirect_uri: Redirect URI after authorization
            state: State parameter for security
            tenant_id: Azure AD tenant ID or "common"
            scopes: OAuth2 scopes to request

        Returns:
            Authorization URL
        """
        default_scopes = [
            "https://graph.microsoft.com/Calendars.ReadWrite",
            "https://graph.microsoft.com/OnlineMeetings.ReadWrite",
            "offline_access",
        ]

        auth_url = cls.AUTH_URL.format(tenant=tenant_id)

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes or default_scopes),
            "state": state,
            "response_mode": "query",
        }

        return f"{auth_url}?{urlencode(params)}"

    @classmethod
    async def exchange_code(
        cls,
        code: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        tenant_id: str = "common",
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback
            client_id: Azure AD application ID
            client_secret: Azure AD client secret
            redirect_uri: Redirect URI used in authorization
            tenant_id: Azure AD tenant ID

        Returns:
            Token response with access_token, refresh_token, etc.
        """
        token_url = cls.TOKEN_URL.format(tenant=tenant_id)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "scope": "https://graph.microsoft.com/.default offline_access",
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
                    "tenant_id": tenant_id,
                }

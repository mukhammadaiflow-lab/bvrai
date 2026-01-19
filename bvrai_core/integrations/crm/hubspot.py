"""
HubSpot CRM Integration

Complete HubSpot integration for syncing contacts, companies,
deals, and logging call engagements.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..base import (
    AuthenticationError,
    AuthType,
    CallLogEntry,
    CRMProvider,
    ExternalContact,
    ExternalDeal,
    IntegrationConfig,
    IntegrationError,
    IntegrationStatus,
    IntegrationType,
    OAuthCredentials,
    RateLimitError,
)


logger = logging.getLogger(__name__)


class HubSpotProvider(CRMProvider):
    """
    HubSpot CRM integration provider.

    Supports:
    - Contact sync
    - Company sync
    - Deal management
    - Engagement (call) logging
    - Timeline events
    - Custom properties
    """

    PROVIDER_NAME = "hubspot"
    PROVIDER_TYPE = IntegrationType.CRM
    AUTH_TYPE = AuthType.OAUTH2

    # HubSpot OAuth endpoints
    AUTH_URL = "https://app.hubspot.com/oauth/authorize"
    TOKEN_URL = "https://api.hubapi.com/oauth/v1/token"
    SCOPES = [
        "crm.objects.contacts.read",
        "crm.objects.contacts.write",
        "crm.objects.deals.read",
        "crm.objects.deals.write",
        "crm.objects.companies.read",
        "crm.objects.companies.write",
        "sales-email-read",
        "timeline",
    ]

    # API base URL
    BASE_URL = "https://api.hubapi.com"

    def __init__(self, config: IntegrationConfig):
        """Initialize HubSpot provider."""
        super().__init__(config)
        self._http_client = None

    async def connect(self) -> bool:
        """Establish connection to HubSpot."""
        try:
            import httpx

            if not isinstance(self.config.credentials, OAuthCredentials):
                raise AuthenticationError("OAuth credentials required")

            # Check if token needs refresh
            if self.config.credentials.is_expired():
                if not await self.refresh_credentials():
                    return False

            # Initialize HTTP client
            self._http_client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.config.credentials.access_token}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )

            # Test connection
            success, error = await self.test_connection()
            if not success:
                raise IntegrationError(error or "Connection test failed")

            self.config.status = IntegrationStatus.CONNECTED
            self.config.last_connected_at = datetime.utcnow()
            self.config.error_message = None

            logger.info(f"Connected to HubSpot for org {self.config.organization_id}")
            return True

        except Exception as e:
            self.config.status = IntegrationStatus.ERROR
            self.config.error_message = str(e)
            logger.error(f"Failed to connect to HubSpot: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from HubSpot."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self.config.status = IntegrationStatus.DISCONNECTED
        logger.info(f"Disconnected from HubSpot for org {self.config.organization_id}")

    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """Test the HubSpot connection."""
        if not self._http_client:
            return (False, "Not connected")

        try:
            response = await self._http_client.get("/crm/v3/objects/contacts?limit=1")
            if response.status_code == 200:
                return (True, None)
            elif response.status_code == 401:
                return (False, "Authentication expired")
            else:
                return (False, f"HTTP {response.status_code}")
        except Exception as e:
            return (False, str(e))

    async def refresh_credentials(self) -> bool:
        """Refresh OAuth credentials."""
        if not isinstance(self.config.credentials, OAuthCredentials):
            return False

        if not self.config.credentials.refresh_token:
            self.config.status = IntegrationStatus.EXPIRED
            return False

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self.config.credentials.refresh_token,
                        "client_id": self.config.metadata.get("client_id"),
                        "client_secret": self.config.metadata.get("client_secret"),
                    },
                )

                if response.status_code != 200:
                    self.config.status = IntegrationStatus.EXPIRED
                    return False

                data = response.json()
                self.config.credentials.access_token = data["access_token"]
                self.config.credentials.refresh_token = data.get(
                    "refresh_token",
                    self.config.credentials.refresh_token,
                )
                self.config.credentials.expires_at = datetime.utcnow() + timedelta(
                    seconds=data.get("expires_in", 21600)
                )

                # Update HTTP client headers
                if self._http_client:
                    self._http_client.headers["Authorization"] = f"Bearer {data['access_token']}"

                logger.info("Refreshed HubSpot credentials")
                return True

        except Exception as e:
            logger.error(f"Failed to refresh HubSpot credentials: {e}")
            self.config.status = IntegrationStatus.ERROR
            return False

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated request to HubSpot."""
        if not self._http_client:
            raise IntegrationError("Not connected to HubSpot")

        # Check token expiry
        if isinstance(self.config.credentials, OAuthCredentials):
            if self.config.credentials.is_expired():
                if not await self.refresh_credentials():
                    raise AuthenticationError("Token refresh failed")

        try:
            response = await self._http_client.request(
                method,
                endpoint,
                json=json,
                params=params,
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                raise RateLimitError(retry_after)

            # Handle auth errors
            if response.status_code == 401:
                if await self.refresh_credentials():
                    response = await self._http_client.request(
                        method,
                        endpoint,
                        json=json,
                        params=params,
                    )
                else:
                    raise AuthenticationError("Authentication failed")

            if response.status_code >= 400:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("message", f"HTTP {response.status_code}")
                raise IntegrationError(f"HubSpot API error: {error_msg}")

            return response.json() if response.text else {}

        except (RateLimitError, AuthenticationError, IntegrationError):
            raise
        except Exception as e:
            raise IntegrationError(f"HubSpot request failed: {e}")

    def _contact_to_external(self, record: Dict[str, Any]) -> ExternalContact:
        """Convert HubSpot Contact to ExternalContact."""
        props = record.get("properties", {})

        return ExternalContact(
            external_id=record.get("id", ""),
            provider=self.PROVIDER_NAME,
            first_name=props.get("firstname"),
            last_name=props.get("lastname"),
            email=props.get("email"),
            phone=props.get("phone"),
            mobile=props.get("mobilephone"),
            company=props.get("company"),
            title=props.get("jobtitle"),
            street=props.get("address"),
            city=props.get("city"),
            state=props.get("state"),
            postal_code=props.get("zip"),
            country=props.get("country"),
            owner_id=props.get("hubspot_owner_id"),
            lead_source=props.get("hs_lead_status"),
            lifecycle_stage=props.get("lifecyclestage"),
            created_at=self._parse_datetime(props.get("createdate")),
            updated_at=self._parse_datetime(props.get("lastmodifieddate")),
            last_activity_at=self._parse_datetime(props.get("notes_last_updated")),
            custom_fields={
                k: v for k, v in props.items()
                if k not in self._standard_contact_fields()
            },
        )

    def _external_to_contact(self, contact: ExternalContact) -> Dict[str, Any]:
        """Convert ExternalContact to HubSpot Contact format."""
        properties = {}

        if contact.first_name:
            properties["firstname"] = contact.first_name
        if contact.last_name:
            properties["lastname"] = contact.last_name
        if contact.email:
            properties["email"] = contact.email
        if contact.phone:
            properties["phone"] = contact.phone
        if contact.mobile:
            properties["mobilephone"] = contact.mobile
        if contact.company:
            properties["company"] = contact.company
        if contact.title:
            properties["jobtitle"] = contact.title
        if contact.street:
            properties["address"] = contact.street
        if contact.city:
            properties["city"] = contact.city
        if contact.state:
            properties["state"] = contact.state
        if contact.postal_code:
            properties["zip"] = contact.postal_code
        if contact.country:
            properties["country"] = contact.country
        if contact.lifecycle_stage:
            properties["lifecyclestage"] = contact.lifecycle_stage

        # Add custom fields
        for key, hs_field in self.config.field_mappings.items():
            if key in contact.custom_fields:
                properties[hs_field] = contact.custom_fields[key]

        return {"properties": properties}

    def _deal_to_external(self, record: Dict[str, Any]) -> ExternalDeal:
        """Convert HubSpot Deal to ExternalDeal."""
        props = record.get("properties", {})

        amount = props.get("amount")
        if amount:
            try:
                amount = float(amount)
            except (ValueError, TypeError):
                amount = None

        return ExternalDeal(
            external_id=record.get("id", ""),
            provider=self.PROVIDER_NAME,
            name=props.get("dealname", ""),
            amount=amount,
            currency=props.get("deal_currency_code", "USD"),
            stage=props.get("dealstage"),
            probability=None,  # HubSpot uses stage-based probability
            owner_id=props.get("hubspot_owner_id"),
            close_date=self._parse_datetime(props.get("closedate")),
            created_at=self._parse_datetime(props.get("createdate")),
            updated_at=self._parse_datetime(props.get("hs_lastmodifieddate")),
            is_won=props.get("hs_is_closed_won") == "true",
            is_closed=props.get("hs_is_closed") == "true",
        )

    def _standard_contact_fields(self) -> set:
        """Get standard HubSpot contact property names."""
        return {
            "firstname", "lastname", "email", "phone", "mobilephone",
            "company", "jobtitle", "address", "city", "state", "zip",
            "country", "hubspot_owner_id", "hs_lead_status", "lifecyclestage",
            "createdate", "lastmodifieddate", "notes_last_updated",
        }

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse HubSpot datetime string."""
        if not value:
            return None
        try:
            # HubSpot uses milliseconds timestamp
            if value.isdigit():
                return datetime.utcfromtimestamp(int(value) / 1000)
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    async def list_contacts(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        updated_since: Optional[datetime] = None,
    ) -> Tuple[List[ExternalContact], Optional[str]]:
        """List contacts from HubSpot."""
        properties = [
            "firstname", "lastname", "email", "phone", "mobilephone",
            "company", "jobtitle", "address", "city", "state", "zip",
            "country", "hubspot_owner_id", "hs_lead_status", "lifecyclestage",
            "createdate", "lastmodifieddate",
        ]

        params = {
            "limit": min(limit, 100),
            "properties": ",".join(properties),
        }

        if cursor:
            params["after"] = cursor

        endpoint = "/crm/v3/objects/contacts"

        # Use search API for filtering by update time
        if updated_since:
            search_data = {
                "filterGroups": [{
                    "filters": [{
                        "propertyName": "lastmodifieddate",
                        "operator": "GTE",
                        "value": str(int(updated_since.timestamp() * 1000)),
                    }]
                }],
                "sorts": [{"propertyName": "lastmodifieddate", "direction": "DESCENDING"}],
                "properties": properties,
                "limit": min(limit, 100),
            }
            if cursor:
                search_data["after"] = cursor

            result = await self._make_request("POST", f"{endpoint}/search", json=search_data)
        else:
            result = await self._make_request("GET", endpoint, params=params)

        contacts = [
            self._contact_to_external(record)
            for record in result.get("results", [])
        ]

        paging = result.get("paging", {})
        next_cursor = paging.get("next", {}).get("after")

        return (contacts, next_cursor)

    async def get_contact(self, external_id: str) -> Optional[ExternalContact]:
        """Get a contact by HubSpot ID."""
        try:
            properties = [
                "firstname", "lastname", "email", "phone", "mobilephone",
                "company", "jobtitle", "address", "city", "state", "zip",
                "country", "hubspot_owner_id", "hs_lead_status", "lifecyclestage",
                "createdate", "lastmodifieddate",
            ]

            result = await self._make_request(
                "GET",
                f"/crm/v3/objects/contacts/{external_id}",
                params={"properties": ",".join(properties)},
            )
            return self._contact_to_external(result)
        except IntegrationError:
            return None

    async def create_contact(self, contact: ExternalContact) -> ExternalContact:
        """Create a contact in HubSpot."""
        data = self._external_to_contact(contact)

        result = await self._make_request(
            "POST",
            "/crm/v3/objects/contacts",
            json=data,
        )

        contact.external_id = result.get("id", "")
        return contact

    async def update_contact(
        self,
        external_id: str,
        updates: Dict[str, Any],
    ) -> ExternalContact:
        """Update a contact in HubSpot."""
        # Map field names
        properties = {}
        field_map = {
            "first_name": "firstname",
            "last_name": "lastname",
            "email": "email",
            "phone": "phone",
            "mobile": "mobilephone",
            "company": "company",
            "title": "jobtitle",
            "street": "address",
            "city": "city",
            "state": "state",
            "postal_code": "zip",
            "country": "country",
        }

        for key, value in updates.items():
            if key in self.config.field_mappings:
                properties[self.config.field_mappings[key]] = value
            elif key in field_map:
                properties[field_map[key]] = value
            else:
                properties[key] = value

        await self._make_request(
            "PATCH",
            f"/crm/v3/objects/contacts/{external_id}",
            json={"properties": properties},
        )

        return await self.get_contact(external_id)

    async def search_contacts(
        self,
        query: str,
        limit: int = 20,
    ) -> List[ExternalContact]:
        """Search contacts in HubSpot."""
        search_data = {
            "query": query,
            "limit": limit,
            "properties": [
                "firstname", "lastname", "email", "phone", "mobilephone",
                "company", "jobtitle",
            ],
        }

        result = await self._make_request(
            "POST",
            "/crm/v3/objects/contacts/search",
            json=search_data,
        )

        return [
            self._contact_to_external(record)
            for record in result.get("results", [])
        ]

    async def log_call(self, entry: CallLogEntry) -> str:
        """Log a call engagement in HubSpot."""
        # Create engagement
        engagement_data = {
            "engagement": {
                "active": True,
                "type": "CALL",
                "timestamp": int(entry.started_at.timestamp() * 1000),
            },
            "metadata": {
                "toNumber": entry.to_number,
                "fromNumber": entry.from_number,
                "status": self._map_call_status(entry.status),
                "durationMilliseconds": entry.duration_seconds * 1000,
                "body": self._build_call_body(entry),
                "disposition": self._map_disposition(entry.outcome),
            },
        }

        # Associate with contact
        if entry.contact_id:
            engagement_data["associations"] = {
                "contactIds": [int(entry.contact_id)] if entry.contact_id.isdigit() else [],
            }

        result = await self._make_request(
            "POST",
            "/engagements/v1/engagements",
            json=engagement_data,
        )

        external_id = str(result.get("engagement", {}).get("id", ""))
        logger.info(f"Logged call {entry.call_id} to HubSpot as engagement {external_id}")

        return external_id

    def _map_call_status(self, status: str) -> str:
        """Map call status to HubSpot status."""
        status_map = {
            "completed": "COMPLETED",
            "missed": "NO_ANSWER",
            "voicemail": "VOICEMAIL",
            "failed": "FAILED",
            "busy": "BUSY",
        }
        return status_map.get(status, "COMPLETED")

    def _map_disposition(self, outcome: Optional[str]) -> Optional[str]:
        """Map outcome to HubSpot disposition."""
        if not outcome:
            return None

        disposition_map = {
            "success": "Connected",
            "callback_requested": "Left voicemail",
            "not_interested": "No answer",
            "voicemail_left": "Left voicemail",
            "no_answer": "No answer",
            "busy": "Busy",
            "wrong_number": "Wrong number",
        }
        return disposition_map.get(outcome, outcome)

    def _build_call_body(self, entry: CallLogEntry) -> str:
        """Build body text for call engagement."""
        lines = [
            f"Call ID: {entry.call_id}",
            f"Direction: {entry.direction}",
            f"Duration: {entry.duration_seconds} seconds",
        ]

        if entry.agent_name:
            lines.append(f"Agent: {entry.agent_name}")

        if entry.outcome:
            lines.append(f"Outcome: {entry.outcome}")

        if entry.transcript_summary:
            lines.append(f"\nSummary:\n{entry.transcript_summary}")

        if entry.notes:
            lines.append(f"\nNotes:\n{entry.notes}")

        if entry.recording_url:
            lines.append(f"\nRecording: {entry.recording_url}")

        return "\n".join(lines)

    async def list_deals(
        self,
        contact_id: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[ExternalDeal], Optional[str]]:
        """List deals from HubSpot."""
        properties = [
            "dealname", "amount", "dealstage", "closedate",
            "hubspot_owner_id", "deal_currency_code",
            "createdate", "hs_lastmodifieddate",
            "hs_is_closed", "hs_is_closed_won",
        ]

        params = {
            "limit": min(limit, 100),
            "properties": ",".join(properties),
        }

        if cursor:
            params["after"] = cursor

        # If contact_id provided, get associated deals
        if contact_id:
            # First get deal associations
            assoc_result = await self._make_request(
                "GET",
                f"/crm/v4/objects/contacts/{contact_id}/associations/deals",
            )

            deal_ids = [
                str(assoc.get("toObjectId"))
                for assoc in assoc_result.get("results", [])
            ]

            if not deal_ids:
                return ([], None)

            # Batch read deals
            result = await self._make_request(
                "POST",
                "/crm/v3/objects/deals/batch/read",
                json={
                    "properties": properties,
                    "inputs": [{"id": did} for did in deal_ids[:100]],
                },
            )

            deals = [
                self._deal_to_external(record)
                for record in result.get("results", [])
            ]

            return (deals, None)
        else:
            result = await self._make_request(
                "GET",
                "/crm/v3/objects/deals",
                params=params,
            )

            deals = [
                self._deal_to_external(record)
                for record in result.get("results", [])
            ]

            paging = result.get("paging", {})
            next_cursor = paging.get("next", {}).get("after")

            return (deals, next_cursor)

    async def create_deal(self, deal: ExternalDeal) -> ExternalDeal:
        """Create a deal in HubSpot."""
        properties = {
            "dealname": deal.name,
            "dealstage": deal.stage or "appointmentscheduled",
        }

        if deal.amount:
            properties["amount"] = str(deal.amount)
        if deal.close_date:
            properties["closedate"] = deal.close_date.strftime("%Y-%m-%d")

        result = await self._make_request(
            "POST",
            "/crm/v3/objects/deals",
            json={"properties": properties},
        )

        deal.external_id = result.get("id", "")

        # Associate with contact if provided
        if deal.contact_id:
            await self._make_request(
                "PUT",
                f"/crm/v4/objects/deals/{deal.external_id}/associations/contacts/{deal.contact_id}",
                json={"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 3},
            )

        return deal

    async def update_deal(
        self,
        deal_id: str,
        updates: Dict[str, Any],
    ) -> ExternalDeal:
        """Update a deal in HubSpot."""
        properties = {}

        # Map common fields to HubSpot properties
        field_mapping = {
            "name": "dealname",
            "stage": "dealstage",
            "amount": "amount",
            "close_date": "closedate",
            "pipeline": "pipeline",
            "probability": "hs_deal_stage_probability",
            "description": "description",
        }

        for key, value in updates.items():
            hs_field = field_mapping.get(key, key)
            if key == "close_date" and value:
                properties[hs_field] = value.strftime("%Y-%m-%d") if hasattr(value, 'strftime') else value
            elif key == "amount":
                properties[hs_field] = str(value) if value else None
            else:
                properties[hs_field] = value

        result = await self._make_request(
            "PATCH",
            f"/crm/v3/objects/deals/{deal_id}",
            json={"properties": properties},
        )

        return self._deal_to_external(result)

    async def find_contact_by_phone(self, phone: str) -> Optional[ExternalContact]:
        """Find a contact by phone number."""
        normalized = phone.replace("+", "").replace("-", "").replace(" ", "").replace("(", "").replace(")", "")

        search_data = {
            "filterGroups": [{
                "filters": [
                    {"propertyName": "phone", "operator": "CONTAINS_TOKEN", "value": normalized[-10:]},
                ]
            }],
            "limit": 1,
            "properties": ["firstname", "lastname", "email", "phone", "mobilephone"],
        }

        result = await self._make_request(
            "POST",
            "/crm/v3/objects/contacts/search",
            json=search_data,
        )

        results = result.get("results", [])
        if results:
            return self._contact_to_external(results[0])

        # Try mobile phone
        search_data["filterGroups"][0]["filters"][0]["propertyName"] = "mobilephone"
        result = await self._make_request(
            "POST",
            "/crm/v3/objects/contacts/search",
            json=search_data,
        )

        results = result.get("results", [])
        if results:
            return self._contact_to_external(results[0])

        return None

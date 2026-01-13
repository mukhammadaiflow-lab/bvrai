"""
Salesforce CRM Integration

Complete Salesforce integration for syncing contacts, accounts,
opportunities, and logging call activities.
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


class SalesforceProvider(CRMProvider):
    """
    Salesforce CRM integration provider.

    Supports:
    - Contact and Lead sync
    - Account sync
    - Opportunity management
    - Task/Activity logging for calls
    - Custom object support
    """

    PROVIDER_NAME = "salesforce"
    PROVIDER_TYPE = IntegrationType.CRM
    AUTH_TYPE = AuthType.OAUTH2

    # Salesforce OAuth endpoints
    AUTH_URL = "https://login.salesforce.com/services/oauth2/authorize"
    TOKEN_URL = "https://login.salesforce.com/services/oauth2/token"
    SCOPES = ["api", "refresh_token", "offline_access"]

    # API version
    API_VERSION = "v59.0"

    def __init__(self, config: IntegrationConfig):
        """Initialize Salesforce provider."""
        super().__init__(config)
        self._instance_url: Optional[str] = None
        self._http_client = None

    async def connect(self) -> bool:
        """Establish connection to Salesforce."""
        try:
            import httpx

            if not isinstance(self.config.credentials, OAuthCredentials):
                raise AuthenticationError("OAuth credentials required")

            # Check if token needs refresh
            if self.config.credentials.is_expired():
                if not await self.refresh_credentials():
                    return False

            # Get instance URL from token response metadata
            self._instance_url = self.config.metadata.get("instance_url")
            if not self._instance_url:
                raise IntegrationError("Instance URL not found in credentials")

            # Initialize HTTP client
            self._http_client = httpx.AsyncClient(
                base_url=f"{self._instance_url}/services/data/{self.API_VERSION}",
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

            logger.info(f"Connected to Salesforce for org {self.config.organization_id}")
            return True

        except Exception as e:
            self.config.status = IntegrationStatus.ERROR
            self.config.error_message = str(e)
            logger.error(f"Failed to connect to Salesforce: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Salesforce."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self.config.status = IntegrationStatus.DISCONNECTED
        logger.info(f"Disconnected from Salesforce for org {self.config.organization_id}")

    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """Test the Salesforce connection."""
        if not self._http_client:
            return (False, "Not connected")

        try:
            response = await self._http_client.get("/limits")
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
                self.config.credentials.expires_at = datetime.utcnow() + timedelta(
                    seconds=data.get("expires_in", 7200)
                )
                self._instance_url = data.get("instance_url", self._instance_url)
                self.config.metadata["instance_url"] = self._instance_url

                # Update HTTP client headers
                if self._http_client:
                    self._http_client.headers["Authorization"] = f"Bearer {data['access_token']}"

                logger.info("Refreshed Salesforce credentials")
                return True

        except Exception as e:
            logger.error(f"Failed to refresh Salesforce credentials: {e}")
            self.config.status = IntegrationStatus.ERROR
            return False

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated request to Salesforce."""
        if not self._http_client:
            raise IntegrationError("Not connected to Salesforce")

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
                retry_after = int(response.headers.get("Retry-After", 60))
                raise RateLimitError(retry_after)

            # Handle auth errors
            if response.status_code == 401:
                if await self.refresh_credentials():
                    # Retry with new token
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
                raise IntegrationError(f"Salesforce API error: {error_msg}")

            return response.json() if response.text else {}

        except (RateLimitError, AuthenticationError, IntegrationError):
            raise
        except Exception as e:
            raise IntegrationError(f"Salesforce request failed: {e}")

    def _contact_to_external(self, record: Dict[str, Any]) -> ExternalContact:
        """Convert Salesforce Contact to ExternalContact."""
        return ExternalContact(
            external_id=record.get("Id", ""),
            provider=self.PROVIDER_NAME,
            first_name=record.get("FirstName"),
            last_name=record.get("LastName"),
            email=record.get("Email"),
            phone=record.get("Phone"),
            mobile=record.get("MobilePhone"),
            company=record.get("Account", {}).get("Name") if record.get("Account") else None,
            title=record.get("Title"),
            street=record.get("MailingStreet"),
            city=record.get("MailingCity"),
            state=record.get("MailingState"),
            postal_code=record.get("MailingPostalCode"),
            country=record.get("MailingCountry"),
            owner_id=record.get("OwnerId"),
            lead_source=record.get("LeadSource"),
            created_at=self._parse_datetime(record.get("CreatedDate")),
            updated_at=self._parse_datetime(record.get("LastModifiedDate")),
            last_activity_at=self._parse_datetime(record.get("LastActivityDate")),
            custom_fields={
                k: v for k, v in record.items()
                if k not in self._standard_contact_fields()
            },
        )

    def _lead_to_external(self, record: Dict[str, Any]) -> ExternalContact:
        """Convert Salesforce Lead to ExternalContact."""
        return ExternalContact(
            external_id=record.get("Id", ""),
            provider=self.PROVIDER_NAME,
            first_name=record.get("FirstName"),
            last_name=record.get("LastName"),
            email=record.get("Email"),
            phone=record.get("Phone"),
            mobile=record.get("MobilePhone"),
            company=record.get("Company"),
            title=record.get("Title"),
            street=record.get("Street"),
            city=record.get("City"),
            state=record.get("State"),
            postal_code=record.get("PostalCode"),
            country=record.get("Country"),
            owner_id=record.get("OwnerId"),
            lead_source=record.get("LeadSource"),
            lifecycle_stage="lead",
            created_at=self._parse_datetime(record.get("CreatedDate")),
            updated_at=self._parse_datetime(record.get("LastModifiedDate")),
            custom_fields={
                "_object_type": "Lead",
                "status": record.get("Status"),
                "rating": record.get("Rating"),
            },
        )

    def _external_to_contact(self, contact: ExternalContact) -> Dict[str, Any]:
        """Convert ExternalContact to Salesforce Contact format."""
        data = {}

        if contact.first_name:
            data["FirstName"] = contact.first_name
        if contact.last_name:
            data["LastName"] = contact.last_name
        if contact.email:
            data["Email"] = contact.email
        if contact.phone:
            data["Phone"] = contact.phone
        if contact.mobile:
            data["MobilePhone"] = contact.mobile
        if contact.title:
            data["Title"] = contact.title
        if contact.street:
            data["MailingStreet"] = contact.street
        if contact.city:
            data["MailingCity"] = contact.city
        if contact.state:
            data["MailingState"] = contact.state
        if contact.postal_code:
            data["MailingPostalCode"] = contact.postal_code
        if contact.country:
            data["MailingCountry"] = contact.country
        if contact.lead_source:
            data["LeadSource"] = contact.lead_source

        # Add custom fields (mapped)
        for key, sf_field in self.config.field_mappings.items():
            if key in contact.custom_fields:
                data[sf_field] = contact.custom_fields[key]

        return data

    def _opportunity_to_deal(self, record: Dict[str, Any]) -> ExternalDeal:
        """Convert Salesforce Opportunity to ExternalDeal."""
        return ExternalDeal(
            external_id=record.get("Id", ""),
            provider=self.PROVIDER_NAME,
            name=record.get("Name", ""),
            amount=record.get("Amount"),
            currency=record.get("CurrencyIsoCode", "USD"),
            stage=record.get("StageName"),
            probability=record.get("Probability"),
            contact_id=record.get("ContactId"),
            company_id=record.get("AccountId"),
            owner_id=record.get("OwnerId"),
            close_date=self._parse_datetime(record.get("CloseDate")),
            created_at=self._parse_datetime(record.get("CreatedDate")),
            updated_at=self._parse_datetime(record.get("LastModifiedDate")),
            is_won=record.get("IsWon", False),
            is_closed=record.get("IsClosed", False),
        )

    def _standard_contact_fields(self) -> set:
        """Get standard Contact field names."""
        return {
            "Id", "FirstName", "LastName", "Email", "Phone", "MobilePhone",
            "Account", "Title", "MailingStreet", "MailingCity", "MailingState",
            "MailingPostalCode", "MailingCountry", "OwnerId", "LeadSource",
            "CreatedDate", "LastModifiedDate", "LastActivityDate",
        }

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse Salesforce datetime string."""
        if not value:
            return None
        try:
            # Handle different formats
            if "T" in value:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            else:
                return datetime.strptime(value, "%Y-%m-%d")
        except (ValueError, TypeError):
            return None

    async def list_contacts(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        updated_since: Optional[datetime] = None,
    ) -> Tuple[List[ExternalContact], Optional[str]]:
        """List contacts from Salesforce."""
        fields = [
            "Id", "FirstName", "LastName", "Email", "Phone", "MobilePhone",
            "Account.Name", "Title", "MailingStreet", "MailingCity",
            "MailingState", "MailingPostalCode", "MailingCountry",
            "OwnerId", "LeadSource", "CreatedDate", "LastModifiedDate",
        ]

        query = f"SELECT {', '.join(fields)} FROM Contact"

        conditions = []
        if updated_since:
            conditions.append(f"LastModifiedDate >= {updated_since.isoformat()}")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += f" ORDER BY LastModifiedDate DESC LIMIT {limit}"

        if cursor:
            endpoint = cursor  # Cursor is the next records URL
        else:
            endpoint = f"/query?q={query}"

        result = await self._make_request("GET", endpoint)

        contacts = [
            self._contact_to_external(record)
            for record in result.get("records", [])
        ]

        next_cursor = result.get("nextRecordsUrl")

        return (contacts, next_cursor)

    async def get_contact(self, external_id: str) -> Optional[ExternalContact]:
        """Get a contact by Salesforce ID."""
        try:
            result = await self._make_request("GET", f"/sobjects/Contact/{external_id}")
            return self._contact_to_external(result)
        except IntegrationError:
            return None

    async def create_contact(self, contact: ExternalContact) -> ExternalContact:
        """Create a contact in Salesforce."""
        data = self._external_to_contact(contact)

        result = await self._make_request("POST", "/sobjects/Contact", json=data)

        contact.external_id = result.get("id", "")
        return contact

    async def update_contact(
        self,
        external_id: str,
        updates: Dict[str, Any],
    ) -> ExternalContact:
        """Update a contact in Salesforce."""
        # Map field names
        sf_updates = {}
        for key, value in updates.items():
            if key in self.config.field_mappings:
                sf_updates[self.config.field_mappings[key]] = value
            else:
                # Convert common field names
                field_map = {
                    "first_name": "FirstName",
                    "last_name": "LastName",
                    "email": "Email",
                    "phone": "Phone",
                    "mobile": "MobilePhone",
                    "title": "Title",
                }
                sf_updates[field_map.get(key, key)] = value

        await self._make_request(
            "PATCH",
            f"/sobjects/Contact/{external_id}",
            json=sf_updates,
        )

        return await self.get_contact(external_id)

    async def search_contacts(
        self,
        query: str,
        limit: int = 20,
    ) -> List[ExternalContact]:
        """Search contacts in Salesforce using SOSL."""
        sosl = f"FIND {{{query}}} IN ALL FIELDS RETURNING Contact(Id, FirstName, LastName, Email, Phone) LIMIT {limit}"

        result = await self._make_request("GET", f"/search?q={sosl}")

        contacts = []
        for record in result.get("searchRecords", []):
            if record.get("attributes", {}).get("type") == "Contact":
                contacts.append(self._contact_to_external(record))

        return contacts

    async def log_call(self, entry: CallLogEntry) -> str:
        """Log a call activity in Salesforce."""
        # Create a Task for the call
        task_data = {
            "Subject": f"Call - {entry.direction.title()}",
            "Status": "Completed",
            "Priority": "Normal",
            "TaskSubtype": "Call",
            "Type": "Call",
            "CallType": "Outbound" if entry.direction == "outbound" else "Inbound",
            "CallDurationInSeconds": entry.duration_seconds,
            "Description": self._build_call_description(entry),
            "ActivityDate": entry.started_at.strftime("%Y-%m-%d"),
        }

        # Associate with contact if available
        if entry.contact_id:
            task_data["WhoId"] = entry.contact_id

        # Associate with deal/opportunity if available
        if entry.deal_id:
            task_data["WhatId"] = entry.deal_id

        result = await self._make_request("POST", "/sobjects/Task", json=task_data)

        external_id = result.get("id", "")
        logger.info(f"Logged call {entry.call_id} to Salesforce as Task {external_id}")

        return external_id

    def _build_call_description(self, entry: CallLogEntry) -> str:
        """Build description text for call log."""
        lines = [
            f"Call ID: {entry.call_id}",
            f"Direction: {entry.direction}",
            f"From: {entry.from_number}",
            f"To: {entry.to_number}",
            f"Duration: {entry.duration_seconds} seconds",
            f"Status: {entry.status}",
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
        """List opportunities from Salesforce."""
        fields = [
            "Id", "Name", "Amount", "StageName", "Probability",
            "ContactId", "AccountId", "OwnerId", "CloseDate",
            "CreatedDate", "LastModifiedDate", "IsWon", "IsClosed",
        ]

        query = f"SELECT {', '.join(fields)} FROM Opportunity"

        if contact_id:
            query += f" WHERE ContactId = '{contact_id}'"

        query += f" ORDER BY LastModifiedDate DESC LIMIT {limit}"

        if cursor:
            endpoint = cursor
        else:
            endpoint = f"/query?q={query}"

        result = await self._make_request("GET", endpoint)

        deals = [
            self._opportunity_to_deal(record)
            for record in result.get("records", [])
        ]

        next_cursor = result.get("nextRecordsUrl")

        return (deals, next_cursor)

    async def create_deal(self, deal: ExternalDeal) -> ExternalDeal:
        """Create an opportunity in Salesforce."""
        data = {
            "Name": deal.name,
            "StageName": deal.stage or "Prospecting",
            "CloseDate": deal.close_date.strftime("%Y-%m-%d") if deal.close_date else None,
        }

        if deal.amount:
            data["Amount"] = deal.amount
        if deal.contact_id:
            data["ContactId"] = deal.contact_id
        if deal.company_id:
            data["AccountId"] = deal.company_id

        result = await self._make_request("POST", "/sobjects/Opportunity", json=data)

        deal.external_id = result.get("id", "")
        return deal

    async def find_contact_by_phone(self, phone: str) -> Optional[ExternalContact]:
        """Find a contact by phone number."""
        # Normalize phone for search
        normalized = phone.replace("+", "").replace("-", "").replace(" ", "").replace("(", "").replace(")", "")

        query = f"""
            SELECT Id, FirstName, LastName, Email, Phone, MobilePhone
            FROM Contact
            WHERE Phone LIKE '%{normalized[-10:]}%'
            OR MobilePhone LIKE '%{normalized[-10:]}%'
            LIMIT 1
        """

        result = await self._make_request("GET", f"/query?q={query}")

        records = result.get("records", [])
        if records:
            return self._contact_to_external(records[0])

        return None

    async def find_lead_by_phone(self, phone: str) -> Optional[ExternalContact]:
        """Find a lead by phone number."""
        normalized = phone.replace("+", "").replace("-", "").replace(" ", "").replace("(", "").replace(")", "")

        query = f"""
            SELECT Id, FirstName, LastName, Email, Phone, MobilePhone, Company, Status
            FROM Lead
            WHERE Phone LIKE '%{normalized[-10:]}%'
            OR MobilePhone LIKE '%{normalized[-10:]}%'
            LIMIT 1
        """

        result = await self._make_request("GET", f"/query?q={query}")

        records = result.get("records", [])
        if records:
            return self._lead_to_external(records[0])

        return None

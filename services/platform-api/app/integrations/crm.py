"""CRM integrations."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
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
class CRMContact:
    """CRM contact."""
    id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get full name."""
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "email": self.email,
            "phone": self.phone,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "company": self.company,
            "title": self.title,
            "custom_fields": self.custom_fields,
        }


@dataclass
class CRMDeal:
    """CRM deal/opportunity."""
    id: str
    name: str
    amount: Optional[float] = None
    stage: Optional[str] = None
    contact_id: Optional[str] = None
    probability: Optional[float] = None
    close_date: Optional[datetime] = None
    created_at: Optional[datetime] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "amount": self.amount,
            "stage": self.stage,
            "contact_id": self.contact_id,
            "probability": self.probability,
            "close_date": self.close_date.isoformat() if self.close_date else None,
            "custom_fields": self.custom_fields,
        }


@dataclass
class CRMActivity:
    """CRM activity/task."""
    id: str
    type: str  # call, email, meeting, task
    subject: str
    description: Optional[str] = None
    contact_id: Optional[str] = None
    deal_id: Optional[str] = None
    due_date: Optional[datetime] = None
    completed: bool = False
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "subject": self.subject,
            "description": self.description,
            "contact_id": self.contact_id,
            "deal_id": self.deal_id,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "completed": self.completed,
        }


class CRMIntegration(Integration):
    """Base class for CRM integrations."""

    @abstractmethod
    async def get_contact_by_phone(self, phone: str) -> Optional[CRMContact]:
        """Get contact by phone number."""
        pass

    @abstractmethod
    async def get_contact_by_email(self, email: str) -> Optional[CRMContact]:
        """Get contact by email."""
        pass

    @abstractmethod
    async def create_contact(self, contact: CRMContact) -> CRMContact:
        """Create a new contact."""
        pass

    @abstractmethod
    async def update_contact(self, contact: CRMContact) -> CRMContact:
        """Update an existing contact."""
        pass

    @abstractmethod
    async def log_call(
        self,
        contact_id: str,
        duration_seconds: int,
        notes: str,
        call_type: str = "inbound",
    ) -> CRMActivity:
        """Log a call activity."""
        pass

    @abstractmethod
    async def search_contacts(
        self,
        query: str,
        limit: int = 10,
    ) -> List[CRMContact]:
        """Search for contacts."""
        pass


class SalesforceIntegration(CRMIntegration):
    """
    Salesforce CRM integration.

    Usage:
        integration = SalesforceIntegration(config)
        await integration.connect(credentials)

        contact = await integration.get_contact_by_phone("+1234567890")
    """

    PROVIDER_NAME = "salesforce"
    REQUIRED_SCOPES = ["api", "refresh_token"]

    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self._instance_url = config.credentials.get("instance_url", "")

    def get_oauth_url(self, redirect_uri: str, state: str) -> str:
        """Get Salesforce OAuth URL."""
        client_id = self.config.settings.get("client_id", "")
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "scope": " ".join(self.REQUIRED_SCOPES),
        }
        return f"https://login.salesforce.com/services/oauth2/authorize?{urllib.parse.urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange code for tokens."""
        client = await self._get_http_client()
        response = await client.post(
            "https://login.salesforce.com/services/oauth2/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self.config.settings.get("client_id"),
                "client_secret": self.config.settings.get("client_secret"),
                "redirect_uri": redirect_uri,
            },
        )
        response.raise_for_status()
        return response.json()

    async def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to Salesforce."""
        self.config.credentials = credentials
        self._instance_url = credentials.get("instance_url", "")
        self.config.status = IntegrationStatus.CONNECTED
        self.config.connected_at = datetime.utcnow()
        return True

    async def disconnect(self) -> bool:
        """Disconnect from Salesforce."""
        self.config.status = IntegrationStatus.DISCONNECTED
        self.config.credentials = {}
        return True

    async def test_connection(self) -> bool:
        """Test Salesforce connection."""
        try:
            await self._make_request(
                "GET",
                f"{self._instance_url}/services/data/v58.0/",
            )
            return True
        except Exception as e:
            logger.error(f"Salesforce connection test failed: {e}")
            return False

    async def refresh_token(self) -> bool:
        """Refresh Salesforce token."""
        try:
            client = await self._get_http_client()
            response = await client.post(
                "https://login.salesforce.com/services/oauth2/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.config.credentials.get("refresh_token"),
                    "client_id": self.config.settings.get("client_id"),
                    "client_secret": self.config.settings.get("client_secret"),
                },
            )
            response.raise_for_status()
            data = response.json()
            self.config.credentials["access_token"] = data["access_token"]
            return True
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False

    async def get_contact_by_phone(self, phone: str) -> Optional[CRMContact]:
        """Get contact by phone."""
        query = f"SELECT Id, Email, Phone, FirstName, LastName, Account.Name, Title FROM Contact WHERE Phone = '{phone}'"
        try:
            data = await self._make_request(
                "GET",
                f"{self._instance_url}/services/data/v58.0/query",
                params={"q": query},
            )
            records = data.get("records", [])
            if records:
                return self._parse_contact(records[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get contact: {e}")
            return None

    async def get_contact_by_email(self, email: str) -> Optional[CRMContact]:
        """Get contact by email."""
        query = f"SELECT Id, Email, Phone, FirstName, LastName, Account.Name, Title FROM Contact WHERE Email = '{email}'"
        try:
            data = await self._make_request(
                "GET",
                f"{self._instance_url}/services/data/v58.0/query",
                params={"q": query},
            )
            records = data.get("records", [])
            if records:
                return self._parse_contact(records[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get contact: {e}")
            return None

    async def create_contact(self, contact: CRMContact) -> CRMContact:
        """Create contact in Salesforce."""
        data = await self._make_request(
            "POST",
            f"{self._instance_url}/services/data/v58.0/sobjects/Contact/",
            json={
                "FirstName": contact.first_name,
                "LastName": contact.last_name,
                "Email": contact.email,
                "Phone": contact.phone,
                "Title": contact.title,
            },
        )
        contact.id = data["id"]
        return contact

    async def update_contact(self, contact: CRMContact) -> CRMContact:
        """Update contact in Salesforce."""
        await self._make_request(
            "PATCH",
            f"{self._instance_url}/services/data/v58.0/sobjects/Contact/{contact.id}",
            json={
                "FirstName": contact.first_name,
                "LastName": contact.last_name,
                "Email": contact.email,
                "Phone": contact.phone,
                "Title": contact.title,
            },
        )
        return contact

    async def log_call(
        self,
        contact_id: str,
        duration_seconds: int,
        notes: str,
        call_type: str = "inbound",
    ) -> CRMActivity:
        """Log call in Salesforce."""
        data = await self._make_request(
            "POST",
            f"{self._instance_url}/services/data/v58.0/sobjects/Task/",
            json={
                "WhoId": contact_id,
                "Subject": f"{call_type.title()} Call",
                "Description": notes,
                "CallDurationInSeconds": duration_seconds,
                "CallType": call_type.title(),
                "Status": "Completed",
                "TaskSubtype": "Call",
            },
        )
        return CRMActivity(
            id=data["id"],
            type="call",
            subject=f"{call_type.title()} Call",
            description=notes,
            contact_id=contact_id,
            completed=True,
        )

    async def search_contacts(self, query: str, limit: int = 10) -> List[CRMContact]:
        """Search contacts in Salesforce."""
        soql = f"SELECT Id, Email, Phone, FirstName, LastName, Account.Name, Title FROM Contact WHERE Name LIKE '%{query}%' LIMIT {limit}"
        try:
            data = await self._make_request(
                "GET",
                f"{self._instance_url}/services/data/v58.0/query",
                params={"q": soql},
            )
            return [self._parse_contact(r) for r in data.get("records", [])]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _parse_contact(self, data: Dict[str, Any]) -> CRMContact:
        """Parse Salesforce contact data."""
        return CRMContact(
            id=data.get("Id", ""),
            email=data.get("Email"),
            phone=data.get("Phone"),
            first_name=data.get("FirstName"),
            last_name=data.get("LastName"),
            company=data.get("Account", {}).get("Name") if data.get("Account") else None,
            title=data.get("Title"),
        )


class HubSpotIntegration(CRMIntegration):
    """
    HubSpot CRM integration.

    Usage:
        integration = HubSpotIntegration(config)
        await integration.connect(credentials)

        contact = await integration.get_contact_by_phone("+1234567890")
    """

    PROVIDER_NAME = "hubspot"
    REQUIRED_SCOPES = ["crm.objects.contacts.read", "crm.objects.contacts.write"]

    BASE_URL = "https://api.hubapi.com"

    def get_oauth_url(self, redirect_uri: str, state: str) -> str:
        """Get HubSpot OAuth URL."""
        client_id = self.config.settings.get("client_id", "")
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "scope": " ".join(self.REQUIRED_SCOPES),
        }
        return f"https://app.hubspot.com/oauth/authorize?{urllib.parse.urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange code for tokens."""
        client = await self._get_http_client()
        response = await client.post(
            "https://api.hubapi.com/oauth/v1/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self.config.settings.get("client_id"),
                "client_secret": self.config.settings.get("client_secret"),
                "redirect_uri": redirect_uri,
            },
        )
        response.raise_for_status()
        return response.json()

    async def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to HubSpot."""
        self.config.credentials = credentials
        self.config.status = IntegrationStatus.CONNECTED
        self.config.connected_at = datetime.utcnow()
        return True

    async def disconnect(self) -> bool:
        """Disconnect from HubSpot."""
        self.config.status = IntegrationStatus.DISCONNECTED
        self.config.credentials = {}
        return True

    async def test_connection(self) -> bool:
        """Test HubSpot connection."""
        try:
            await self._make_request(
                "GET",
                f"{self.BASE_URL}/crm/v3/objects/contacts",
                params={"limit": 1},
            )
            return True
        except Exception as e:
            logger.error(f"HubSpot connection test failed: {e}")
            return False

    async def refresh_token(self) -> bool:
        """Refresh HubSpot token."""
        try:
            client = await self._get_http_client()
            response = await client.post(
                "https://api.hubapi.com/oauth/v1/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.config.credentials.get("refresh_token"),
                    "client_id": self.config.settings.get("client_id"),
                    "client_secret": self.config.settings.get("client_secret"),
                },
            )
            response.raise_for_status()
            data = response.json()
            self.config.credentials["access_token"] = data["access_token"]
            self.config.credentials["refresh_token"] = data["refresh_token"]
            return True
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False

    async def get_contact_by_phone(self, phone: str) -> Optional[CRMContact]:
        """Get contact by phone."""
        try:
            data = await self._make_request(
                "POST",
                f"{self.BASE_URL}/crm/v3/objects/contacts/search",
                json={
                    "filterGroups": [{
                        "filters": [{
                            "propertyName": "phone",
                            "operator": "EQ",
                            "value": phone,
                        }]
                    }],
                    "properties": ["email", "phone", "firstname", "lastname", "company", "jobtitle"],
                },
            )
            results = data.get("results", [])
            if results:
                return self._parse_contact(results[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get contact: {e}")
            return None

    async def get_contact_by_email(self, email: str) -> Optional[CRMContact]:
        """Get contact by email."""
        try:
            data = await self._make_request(
                "POST",
                f"{self.BASE_URL}/crm/v3/objects/contacts/search",
                json={
                    "filterGroups": [{
                        "filters": [{
                            "propertyName": "email",
                            "operator": "EQ",
                            "value": email,
                        }]
                    }],
                    "properties": ["email", "phone", "firstname", "lastname", "company", "jobtitle"],
                },
            )
            results = data.get("results", [])
            if results:
                return self._parse_contact(results[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get contact: {e}")
            return None

    async def create_contact(self, contact: CRMContact) -> CRMContact:
        """Create contact in HubSpot."""
        data = await self._make_request(
            "POST",
            f"{self.BASE_URL}/crm/v3/objects/contacts",
            json={
                "properties": {
                    "email": contact.email,
                    "phone": contact.phone,
                    "firstname": contact.first_name,
                    "lastname": contact.last_name,
                    "company": contact.company,
                    "jobtitle": contact.title,
                },
            },
        )
        contact.id = data["id"]
        return contact

    async def update_contact(self, contact: CRMContact) -> CRMContact:
        """Update contact in HubSpot."""
        await self._make_request(
            "PATCH",
            f"{self.BASE_URL}/crm/v3/objects/contacts/{contact.id}",
            json={
                "properties": {
                    "email": contact.email,
                    "phone": contact.phone,
                    "firstname": contact.first_name,
                    "lastname": contact.last_name,
                    "company": contact.company,
                    "jobtitle": contact.title,
                },
            },
        )
        return contact

    async def log_call(
        self,
        contact_id: str,
        duration_seconds: int,
        notes: str,
        call_type: str = "inbound",
    ) -> CRMActivity:
        """Log call in HubSpot."""
        # Create engagement
        data = await self._make_request(
            "POST",
            f"{self.BASE_URL}/crm/v3/objects/calls",
            json={
                "properties": {
                    "hs_call_body": notes,
                    "hs_call_duration": str(duration_seconds * 1000),  # milliseconds
                    "hs_call_direction": call_type.upper(),
                    "hs_call_status": "COMPLETED",
                },
            },
        )

        call_id = data["id"]

        # Associate with contact
        await self._make_request(
            "PUT",
            f"{self.BASE_URL}/crm/v3/objects/calls/{call_id}/associations/contacts/{contact_id}/call_to_contact",
        )

        return CRMActivity(
            id=call_id,
            type="call",
            subject=f"{call_type.title()} Call",
            description=notes,
            contact_id=contact_id,
            completed=True,
        )

    async def search_contacts(self, query: str, limit: int = 10) -> List[CRMContact]:
        """Search contacts in HubSpot."""
        try:
            data = await self._make_request(
                "POST",
                f"{self.BASE_URL}/crm/v3/objects/contacts/search",
                json={
                    "query": query,
                    "limit": limit,
                    "properties": ["email", "phone", "firstname", "lastname", "company", "jobtitle"],
                },
            )
            return [self._parse_contact(r) for r in data.get("results", [])]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _parse_contact(self, data: Dict[str, Any]) -> CRMContact:
        """Parse HubSpot contact data."""
        props = data.get("properties", {})
        return CRMContact(
            id=data.get("id", ""),
            email=props.get("email"),
            phone=props.get("phone"),
            first_name=props.get("firstname"),
            last_name=props.get("lastname"),
            company=props.get("company"),
            title=props.get("jobtitle"),
        )

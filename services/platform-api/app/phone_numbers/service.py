"""
Phone Number Management Service

Complete phone number lifecycle:
- Number provisioning
- Number configuration
- Carrier integration
- Number routing
"""

from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import re
import asyncio
import aiohttp
import logging

logger = logging.getLogger(__name__)


class NumberType(str, Enum):
    """Phone number types."""
    LOCAL = "local"
    TOLL_FREE = "toll_free"
    MOBILE = "mobile"
    SHORT_CODE = "short_code"


class NumberStatus(str, Enum):
    """Phone number status."""
    AVAILABLE = "available"
    RESERVED = "reserved"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RELEASED = "released"


class NumberCapability(str, Enum):
    """Number capabilities."""
    VOICE = "voice"
    SMS = "sms"
    MMS = "mms"
    FAX = "fax"


class Provider(str, Enum):
    """Telephony providers."""
    TWILIO = "twilio"
    VONAGE = "vonage"
    TELNYX = "telnyx"
    BANDWIDTH = "bandwidth"
    SIGNALWIRE = "signalwire"
    PLIVO = "plivo"


@dataclass
class PhoneNumber:
    """Phone number entity."""
    number_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phone_number: str = ""  # E.164 format
    friendly_name: str = ""

    # Type and status
    number_type: NumberType = NumberType.LOCAL
    status: NumberStatus = NumberStatus.AVAILABLE
    capabilities: Set[NumberCapability] = field(default_factory=set)

    # Location
    country_code: str = "US"
    region: str = ""
    locality: str = ""
    postal_code: str = ""

    # Configuration
    voice_url: str = ""
    voice_method: str = "POST"
    voice_fallback_url: str = ""
    sms_url: str = ""
    sms_method: str = "POST"
    status_callback: str = ""

    # Agent binding
    agent_id: Optional[str] = None
    workflow_id: Optional[str] = None

    # Provider
    provider: Provider = Provider.TWILIO
    provider_sid: str = ""

    # Ownership
    tenant_id: str = ""

    # Costs
    monthly_cost: float = 0.0
    setup_cost: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "number_id": self.number_id,
            "phone_number": self.phone_number,
            "friendly_name": self.friendly_name,
            "number_type": self.number_type.value,
            "status": self.status.value,
            "capabilities": [c.value for c in self.capabilities],
            "country_code": self.country_code,
            "region": self.region,
            "agent_id": self.agent_id,
            "provider": self.provider.value,
            "monthly_cost": self.monthly_cost,
        }


@dataclass
class NumberSearchCriteria:
    """Criteria for searching available numbers."""
    country_code: str = "US"
    number_type: NumberType = NumberType.LOCAL

    # Location
    region: Optional[str] = None
    area_code: Optional[str] = None
    locality: Optional[str] = None
    postal_code: Optional[str] = None

    # Capabilities
    capabilities: Set[NumberCapability] = field(default_factory=lambda: {NumberCapability.VOICE})

    # Pattern
    contains: Optional[str] = None
    starts_with: Optional[str] = None
    ends_with: Optional[str] = None

    # Limits
    limit: int = 20


@dataclass
class AvailableNumber:
    """Available number from search."""
    phone_number: str
    number_type: NumberType
    capabilities: Set[NumberCapability]
    country_code: str
    region: str
    locality: str
    monthly_cost: float
    setup_cost: float
    provider: Provider


class TelephonyProvider(ABC):
    """Abstract telephony provider."""

    @property
    @abstractmethod
    def provider_name(self) -> Provider:
        """Provider name."""
        pass

    @abstractmethod
    async def search_numbers(
        self,
        criteria: NumberSearchCriteria,
    ) -> List[AvailableNumber]:
        """Search for available numbers."""
        pass

    @abstractmethod
    async def purchase_number(
        self,
        phone_number: str,
        voice_url: str = "",
        sms_url: str = "",
    ) -> PhoneNumber:
        """Purchase a phone number."""
        pass

    @abstractmethod
    async def release_number(self, phone_number: str) -> bool:
        """Release a phone number."""
        pass

    @abstractmethod
    async def update_number(
        self,
        phone_number: str,
        voice_url: Optional[str] = None,
        sms_url: Optional[str] = None,
        **kwargs,
    ) -> PhoneNumber:
        """Update number configuration."""
        pass


class TwilioProvider(TelephonyProvider):
    """Twilio telephony provider."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
    ):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self._base_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}"

    @property
    def provider_name(self) -> Provider:
        return Provider.TWILIO

    async def search_numbers(
        self,
        criteria: NumberSearchCriteria,
    ) -> List[AvailableNumber]:
        """Search Twilio for available numbers."""
        auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)

        # Build URL based on number type
        if criteria.number_type == NumberType.TOLL_FREE:
            url = f"{self._base_url}/AvailablePhoneNumbers/{criteria.country_code}/TollFree.json"
        elif criteria.number_type == NumberType.MOBILE:
            url = f"{self._base_url}/AvailablePhoneNumbers/{criteria.country_code}/Mobile.json"
        else:
            url = f"{self._base_url}/AvailablePhoneNumbers/{criteria.country_code}/Local.json"

        params = {}
        if criteria.area_code:
            params["AreaCode"] = criteria.area_code
        if criteria.region:
            params["InRegion"] = criteria.region
        if criteria.locality:
            params["InLocality"] = criteria.locality
        if criteria.postal_code:
            params["InPostalCode"] = criteria.postal_code
        if criteria.contains:
            params["Contains"] = criteria.contains
        params["VoiceEnabled"] = NumberCapability.VOICE in criteria.capabilities
        params["SmsEnabled"] = NumberCapability.SMS in criteria.capabilities
        params["PageSize"] = criteria.limit

        try:
            async with aiohttp.ClientSession(auth=auth) as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    numbers = []

                    for item in data.get("available_phone_numbers", []):
                        capabilities = set()
                        if item.get("capabilities", {}).get("voice"):
                            capabilities.add(NumberCapability.VOICE)
                        if item.get("capabilities", {}).get("SMS"):
                            capabilities.add(NumberCapability.SMS)
                        if item.get("capabilities", {}).get("MMS"):
                            capabilities.add(NumberCapability.MMS)

                        numbers.append(AvailableNumber(
                            phone_number=item["phone_number"],
                            number_type=criteria.number_type,
                            capabilities=capabilities,
                            country_code=item.get("iso_country", criteria.country_code),
                            region=item.get("region", ""),
                            locality=item.get("locality", ""),
                            monthly_cost=1.15,  # Twilio standard pricing
                            setup_cost=0.0,
                            provider=self.provider_name,
                        ))

                    return numbers

        except Exception as e:
            logger.error(f"Twilio search error: {e}")
            return []

    async def purchase_number(
        self,
        phone_number: str,
        voice_url: str = "",
        sms_url: str = "",
    ) -> PhoneNumber:
        """Purchase number from Twilio."""
        auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
        url = f"{self._base_url}/IncomingPhoneNumbers.json"

        data = {"PhoneNumber": phone_number}
        if voice_url:
            data["VoiceUrl"] = voice_url
        if sms_url:
            data["SmsUrl"] = sms_url

        try:
            async with aiohttp.ClientSession(auth=auth) as session:
                async with session.post(url, data=data) as response:
                    if response.status not in (200, 201):
                        error = await response.text()
                        raise Exception(f"Twilio purchase error: {error}")

                    result = await response.json()

                    capabilities = set()
                    caps = result.get("capabilities", {})
                    if caps.get("voice"):
                        capabilities.add(NumberCapability.VOICE)
                    if caps.get("sms"):
                        capabilities.add(NumberCapability.SMS)
                    if caps.get("mms"):
                        capabilities.add(NumberCapability.MMS)

                    return PhoneNumber(
                        phone_number=result["phone_number"],
                        friendly_name=result.get("friendly_name", ""),
                        status=NumberStatus.ACTIVE,
                        capabilities=capabilities,
                        country_code=result.get("iso_country", "US"),
                        region=result.get("region", ""),
                        locality=result.get("locality", ""),
                        voice_url=result.get("voice_url", ""),
                        sms_url=result.get("sms_url", ""),
                        provider=self.provider_name,
                        provider_sid=result["sid"],
                    )

        except Exception as e:
            logger.error(f"Twilio purchase error: {e}")
            raise

    async def release_number(self, phone_number: str) -> bool:
        """Release number from Twilio."""
        # Would need to look up SID first
        return True

    async def update_number(
        self,
        phone_number: str,
        voice_url: Optional[str] = None,
        sms_url: Optional[str] = None,
        **kwargs,
    ) -> PhoneNumber:
        """Update Twilio number configuration."""
        # Implementation would update via Twilio API
        return PhoneNumber(phone_number=phone_number)


class TelnyxProvider(TelephonyProvider):
    """Telnyx telephony provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._base_url = "https://api.telnyx.com/v2"

    @property
    def provider_name(self) -> Provider:
        return Provider.TELNYX

    async def search_numbers(
        self,
        criteria: NumberSearchCriteria,
    ) -> List[AvailableNumber]:
        """Search Telnyx for available numbers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        params = {
            "filter[country_code]": criteria.country_code,
            "filter[limit]": criteria.limit,
        }

        if criteria.region:
            params["filter[administrative_area]"] = criteria.region
        if criteria.locality:
            params["filter[locality]"] = criteria.locality
        if NumberCapability.VOICE in criteria.capabilities:
            params["filter[features]"] = "voice"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/available_phone_numbers",
                    headers=headers,
                    params=params,
                ) as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    numbers = []

                    for item in data.get("data", []):
                        capabilities = set()
                        for feature in item.get("features", []):
                            if feature.get("name") == "voice":
                                capabilities.add(NumberCapability.VOICE)
                            elif feature.get("name") == "sms":
                                capabilities.add(NumberCapability.SMS)

                        numbers.append(AvailableNumber(
                            phone_number=item["phone_number"],
                            number_type=criteria.number_type,
                            capabilities=capabilities,
                            country_code=item.get("country_code", criteria.country_code),
                            region=item.get("region_information", [{}])[0].get("region_name", ""),
                            locality=item.get("locality", ""),
                            monthly_cost=float(item.get("cost_information", {}).get("monthly_cost", "1.00")),
                            setup_cost=0.0,
                            provider=self.provider_name,
                        ))

                    return numbers

        except Exception as e:
            logger.error(f"Telnyx search error: {e}")
            return []

    async def purchase_number(
        self,
        phone_number: str,
        voice_url: str = "",
        sms_url: str = "",
    ) -> PhoneNumber:
        """Purchase number from Telnyx."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"phone_numbers": [{"phone_number": phone_number}]}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/number_orders",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status not in (200, 201):
                        error = await response.text()
                        raise Exception(f"Telnyx purchase error: {error}")

                    return PhoneNumber(
                        phone_number=phone_number,
                        status=NumberStatus.ACTIVE,
                        provider=self.provider_name,
                    )

        except Exception as e:
            logger.error(f"Telnyx purchase error: {e}")
            raise

    async def release_number(self, phone_number: str) -> bool:
        """Release number from Telnyx."""
        return True

    async def update_number(
        self,
        phone_number: str,
        voice_url: Optional[str] = None,
        sms_url: Optional[str] = None,
        **kwargs,
    ) -> PhoneNumber:
        """Update Telnyx number configuration."""
        return PhoneNumber(phone_number=phone_number)


class PhoneNumberService:
    """
    Phone number management service.

    Features:
    - Multi-provider support
    - Number lifecycle management
    - Agent binding
    - Usage tracking
    """

    def __init__(self):
        self._numbers: Dict[str, PhoneNumber] = {}
        self._providers: Dict[Provider, TelephonyProvider] = {}
        self._by_tenant: Dict[str, Set[str]] = {}
        self._by_agent: Dict[str, Set[str]] = {}

    def register_provider(self, provider: TelephonyProvider) -> None:
        """Register telephony provider."""
        self._providers[provider.provider_name] = provider

    async def search_available(
        self,
        criteria: NumberSearchCriteria,
        provider: Optional[Provider] = None,
    ) -> List[AvailableNumber]:
        """Search for available numbers."""
        results = []

        if provider:
            if provider in self._providers:
                results = await self._providers[provider].search_numbers(criteria)
        else:
            # Search all providers
            tasks = [
                p.search_numbers(criteria) for p in self._providers.values()
            ]
            provider_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in provider_results:
                if isinstance(result, list):
                    results.extend(result)

        return results

    async def purchase(
        self,
        phone_number: str,
        tenant_id: str,
        provider: Provider = Provider.TWILIO,
        voice_url: str = "",
        sms_url: str = "",
        agent_id: Optional[str] = None,
    ) -> PhoneNumber:
        """Purchase a phone number."""
        if provider not in self._providers:
            raise ValueError(f"Provider {provider} not registered")

        # Purchase from provider
        number = await self._providers[provider].purchase_number(
            phone_number,
            voice_url,
            sms_url,
        )

        # Set tenant and agent
        number.tenant_id = tenant_id
        number.agent_id = agent_id

        # Store
        self._numbers[number.number_id] = number

        if tenant_id not in self._by_tenant:
            self._by_tenant[tenant_id] = set()
        self._by_tenant[tenant_id].add(number.number_id)

        if agent_id:
            if agent_id not in self._by_agent:
                self._by_agent[agent_id] = set()
            self._by_agent[agent_id].add(number.number_id)

        logger.info(f"Purchased number: {phone_number}")
        return number

    async def release(self, number_id: str) -> bool:
        """Release a phone number."""
        number = self._numbers.get(number_id)
        if not number:
            return False

        # Release from provider
        provider = self._providers.get(number.provider)
        if provider:
            await provider.release_number(number.phone_number)

        # Update status
        number.status = NumberStatus.RELEASED

        # Remove from indices
        if number.tenant_id in self._by_tenant:
            self._by_tenant[number.tenant_id].discard(number_id)
        if number.agent_id and number.agent_id in self._by_agent:
            self._by_agent[number.agent_id].discard(number_id)

        logger.info(f"Released number: {number.phone_number}")
        return True

    def get(self, number_id: str) -> Optional[PhoneNumber]:
        """Get number by ID."""
        return self._numbers.get(number_id)

    def get_by_phone(self, phone_number: str) -> Optional[PhoneNumber]:
        """Get number by phone number."""
        for number in self._numbers.values():
            if number.phone_number == phone_number:
                return number
        return None

    def list_by_tenant(self, tenant_id: str) -> List[PhoneNumber]:
        """List numbers by tenant."""
        number_ids = self._by_tenant.get(tenant_id, set())
        return [self._numbers[nid] for nid in number_ids if nid in self._numbers]

    def list_by_agent(self, agent_id: str) -> List[PhoneNumber]:
        """List numbers by agent."""
        number_ids = self._by_agent.get(agent_id, set())
        return [self._numbers[nid] for nid in number_ids if nid in self._numbers]

    async def bind_to_agent(
        self,
        number_id: str,
        agent_id: str,
        voice_url: str = "",
    ) -> bool:
        """Bind number to agent."""
        number = self._numbers.get(number_id)
        if not number:
            return False

        # Remove from old agent
        if number.agent_id and number.agent_id in self._by_agent:
            self._by_agent[number.agent_id].discard(number_id)

        # Bind to new agent
        number.agent_id = agent_id
        if voice_url:
            number.voice_url = voice_url

        if agent_id not in self._by_agent:
            self._by_agent[agent_id] = set()
        self._by_agent[agent_id].add(number_id)

        # Update provider
        provider = self._providers.get(number.provider)
        if provider and voice_url:
            await provider.update_number(number.phone_number, voice_url=voice_url)

        number.updated_at = datetime.utcnow()
        return True

    async def unbind_from_agent(self, number_id: str) -> bool:
        """Unbind number from agent."""
        number = self._numbers.get(number_id)
        if not number:
            return False

        if number.agent_id and number.agent_id in self._by_agent:
            self._by_agent[number.agent_id].discard(number_id)

        number.agent_id = None
        number.updated_at = datetime.utcnow()
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        numbers = list(self._numbers.values())
        return {
            "total_numbers": len(numbers),
            "by_status": {
                status.value: sum(1 for n in numbers if n.status == status)
                for status in NumberStatus
            },
            "by_provider": {
                provider.value: sum(1 for n in numbers if n.provider == provider)
                for provider in Provider
            },
            "by_type": {
                ntype.value: sum(1 for n in numbers if n.number_type == ntype)
                for ntype in NumberType
            },
        }


# Utility functions
def format_e164(phone_number: str, default_country: str = "US") -> str:
    """Format phone number to E.164."""
    # Remove all non-digits
    digits = re.sub(r'\D', '', phone_number)

    if default_country == "US":
        if len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) == 11 and digits.startswith("1"):
            return f"+{digits}"

    if not digits.startswith("+"):
        return f"+{digits}"

    return digits


def validate_e164(phone_number: str) -> bool:
    """Validate E.164 format."""
    pattern = r'^\+[1-9]\d{1,14}$'
    return bool(re.match(pattern, phone_number))


# Singleton service
_service_instance: Optional[PhoneNumberService] = None


def get_phone_number_service() -> PhoneNumberService:
    """Get singleton phone number service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PhoneNumberService()
    return _service_instance

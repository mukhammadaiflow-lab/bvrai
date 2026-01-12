"""Phone Numbers API for Builder Engine."""

from typing import Optional, Dict, List, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from bvrai_sdk.client import BvraiClient


class PhoneNumberType(str, Enum):
    """Phone number types."""
    LOCAL = "local"
    TOLL_FREE = "toll_free"
    MOBILE = "mobile"


class PhoneNumberStatus(str, Enum):
    """Phone number status."""
    ACTIVE = "active"
    PENDING = "pending"
    SUSPENDED = "suspended"
    RELEASED = "released"


class PhoneNumberCapability(str, Enum):
    """Phone number capabilities."""
    VOICE = "voice"
    SMS = "sms"
    MMS = "mms"
    FAX = "fax"


@dataclass
class PhoneNumber:
    """A phone number in the system."""
    id: str
    number: str
    friendly_name: Optional[str]
    type: PhoneNumberType
    status: PhoneNumberStatus
    capabilities: List[PhoneNumberCapability]
    agent_id: Optional[str]
    country_code: str
    area_code: Optional[str]
    provider: str
    monthly_cost: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhoneNumber":
        return cls(
            id=data["id"],
            number=data["number"],
            friendly_name=data.get("friendly_name"),
            type=PhoneNumberType(data.get("type", "local")),
            status=PhoneNumberStatus(data.get("status", "active")),
            capabilities=[PhoneNumberCapability(c) for c in data.get("capabilities", ["voice"])],
            agent_id=data.get("agent_id"),
            country_code=data.get("country_code", "US"),
            area_code=data.get("area_code"),
            provider=data.get("provider", "twilio"),
            monthly_cost=data.get("monthly_cost", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class AvailableNumber:
    """An available phone number for purchase."""
    number: str
    friendly_name: str
    type: PhoneNumberType
    capabilities: List[PhoneNumberCapability]
    country_code: str
    area_code: Optional[str]
    region: Optional[str]
    locality: Optional[str]
    monthly_cost: float
    setup_cost: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AvailableNumber":
        return cls(
            number=data["number"],
            friendly_name=data.get("friendly_name", data["number"]),
            type=PhoneNumberType(data.get("type", "local")),
            capabilities=[PhoneNumberCapability(c) for c in data.get("capabilities", ["voice"])],
            country_code=data.get("country_code", "US"),
            area_code=data.get("area_code"),
            region=data.get("region"),
            locality=data.get("locality"),
            monthly_cost=data.get("monthly_cost", 0.0),
            setup_cost=data.get("setup_cost", 0.0),
        )


@dataclass
class PhoneNumberConfig:
    """Configuration for a phone number."""
    voice_url: Optional[str] = None
    voice_fallback_url: Optional[str] = None
    status_callback_url: Optional[str] = None
    voice_method: str = "POST"
    sms_url: Optional[str] = None
    sms_fallback_url: Optional[str] = None
    recording_enabled: bool = True
    transcription_enabled: bool = True
    caller_id_lookup: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voice_url": self.voice_url,
            "voice_fallback_url": self.voice_fallback_url,
            "status_callback_url": self.status_callback_url,
            "voice_method": self.voice_method,
            "sms_url": self.sms_url,
            "sms_fallback_url": self.sms_fallback_url,
            "recording_enabled": self.recording_enabled,
            "transcription_enabled": self.transcription_enabled,
            "caller_id_lookup": self.caller_id_lookup,
        }


class PhoneNumbersAPI:
    """
    Phone Numbers API client.

    Manage phone numbers for inbound and outbound calls.
    """

    def __init__(self, client: "BvraiClient"):
        self._client = client

    # Number Management

    async def list(
        self,
        agent_id: Optional[str] = None,
        status: Optional[PhoneNumberStatus] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[PhoneNumber]:
        """List all phone numbers."""
        params = {"limit": limit, "offset": offset}
        if agent_id:
            params["agent_id"] = agent_id
        if status:
            params["status"] = status.value

        response = await self._client.get("/v1/phone-numbers", params=params)
        return [PhoneNumber.from_dict(n) for n in response.get("phone_numbers", [])]

    async def get(self, number_id: str) -> PhoneNumber:
        """Get a phone number by ID."""
        response = await self._client.get(f"/v1/phone-numbers/{number_id}")
        return PhoneNumber.from_dict(response)

    async def get_by_number(self, phone_number: str) -> PhoneNumber:
        """Get a phone number by the actual number."""
        response = await self._client.get(f"/v1/phone-numbers/lookup/{phone_number}")
        return PhoneNumber.from_dict(response)

    async def update(
        self,
        number_id: str,
        friendly_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        config: Optional[PhoneNumberConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PhoneNumber:
        """Update a phone number."""
        data = {}
        if friendly_name is not None:
            data["friendly_name"] = friendly_name
        if agent_id is not None:
            data["agent_id"] = agent_id
        if config is not None:
            data["config"] = config.to_dict()
        if metadata is not None:
            data["metadata"] = metadata

        response = await self._client.patch(f"/v1/phone-numbers/{number_id}", data=data)
        return PhoneNumber.from_dict(response)

    async def release(self, number_id: str) -> bool:
        """Release a phone number."""
        await self._client.delete(f"/v1/phone-numbers/{number_id}")
        return True

    # Number Purchase

    async def search_available(
        self,
        country_code: str = "US",
        type: PhoneNumberType = PhoneNumberType.LOCAL,
        area_code: Optional[str] = None,
        contains: Optional[str] = None,
        capabilities: Optional[List[PhoneNumberCapability]] = None,
        limit: int = 20,
    ) -> List[AvailableNumber]:
        """Search for available phone numbers to purchase."""
        params = {
            "country_code": country_code,
            "type": type.value,
            "limit": limit,
        }
        if area_code:
            params["area_code"] = area_code
        if contains:
            params["contains"] = contains
        if capabilities:
            params["capabilities"] = ",".join(c.value for c in capabilities)

        response = await self._client.get("/v1/phone-numbers/available", params=params)
        return [AvailableNumber.from_dict(n) for n in response.get("available_numbers", [])]

    async def purchase(
        self,
        phone_number: str,
        friendly_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        config: Optional[PhoneNumberConfig] = None,
    ) -> PhoneNumber:
        """Purchase an available phone number."""
        data = {
            "phone_number": phone_number,
            "friendly_name": friendly_name,
            "agent_id": agent_id,
        }
        if config:
            data["config"] = config.to_dict()

        response = await self._client.post("/v1/phone-numbers/purchase", data=data)
        return PhoneNumber.from_dict(response)

    async def import_number(
        self,
        phone_number: str,
        provider: str,
        provider_sid: str,
        friendly_name: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> PhoneNumber:
        """Import an existing phone number from a provider."""
        data = {
            "phone_number": phone_number,
            "provider": provider,
            "provider_sid": provider_sid,
            "friendly_name": friendly_name,
            "agent_id": agent_id,
        }
        response = await self._client.post("/v1/phone-numbers/import", data=data)
        return PhoneNumber.from_dict(response)

    # Agent Assignment

    async def assign_to_agent(self, number_id: str, agent_id: str) -> PhoneNumber:
        """Assign a phone number to an agent."""
        return await self.update(number_id, agent_id=agent_id)

    async def unassign_from_agent(self, number_id: str) -> PhoneNumber:
        """Unassign a phone number from its agent."""
        data = {"agent_id": None}
        response = await self._client.patch(f"/v1/phone-numbers/{number_id}", data=data)
        return PhoneNumber.from_dict(response)

    async def get_agent_numbers(self, agent_id: str) -> List[PhoneNumber]:
        """Get all phone numbers assigned to an agent."""
        return await self.list(agent_id=agent_id)

    # Configuration

    async def get_config(self, number_id: str) -> PhoneNumberConfig:
        """Get the configuration for a phone number."""
        response = await self._client.get(f"/v1/phone-numbers/{number_id}/config")
        return PhoneNumberConfig(
            voice_url=response.get("voice_url"),
            voice_fallback_url=response.get("voice_fallback_url"),
            status_callback_url=response.get("status_callback_url"),
            voice_method=response.get("voice_method", "POST"),
            sms_url=response.get("sms_url"),
            sms_fallback_url=response.get("sms_fallback_url"),
            recording_enabled=response.get("recording_enabled", True),
            transcription_enabled=response.get("transcription_enabled", True),
            caller_id_lookup=response.get("caller_id_lookup", False),
        )

    async def update_config(
        self,
        number_id: str,
        config: PhoneNumberConfig,
    ) -> PhoneNumberConfig:
        """Update the configuration for a phone number."""
        response = await self._client.put(
            f"/v1/phone-numbers/{number_id}/config",
            data=config.to_dict(),
        )
        return PhoneNumberConfig(
            voice_url=response.get("voice_url"),
            voice_fallback_url=response.get("voice_fallback_url"),
            status_callback_url=response.get("status_callback_url"),
            voice_method=response.get("voice_method", "POST"),
            sms_url=response.get("sms_url"),
            sms_fallback_url=response.get("sms_fallback_url"),
            recording_enabled=response.get("recording_enabled", True),
            transcription_enabled=response.get("transcription_enabled", True),
            caller_id_lookup=response.get("caller_id_lookup", False),
        )

    # Verification

    async def verify_caller_id(
        self,
        phone_number: str,
        friendly_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start caller ID verification for outbound calls."""
        data = {
            "phone_number": phone_number,
            "friendly_name": friendly_name,
        }
        return await self._client.post("/v1/phone-numbers/verify", data=data)

    async def check_verification(self, verification_sid: str) -> Dict[str, Any]:
        """Check the status of a caller ID verification."""
        return await self._client.get(f"/v1/phone-numbers/verify/{verification_sid}")

    async def submit_verification_code(
        self,
        verification_sid: str,
        code: str,
    ) -> Dict[str, Any]:
        """Submit the verification code."""
        data = {"code": code}
        return await self._client.post(
            f"/v1/phone-numbers/verify/{verification_sid}/submit",
            data=data,
        )

    # Convenience Methods

    async def get_active_numbers(self) -> List[PhoneNumber]:
        """Get all active phone numbers."""
        return await self.list(status=PhoneNumberStatus.ACTIVE)

    async def search_local(
        self,
        area_code: str,
        limit: int = 20,
    ) -> List[AvailableNumber]:
        """Search for local numbers in an area code."""
        return await self.search_available(
            type=PhoneNumberType.LOCAL,
            area_code=area_code,
            limit=limit,
        )

    async def search_toll_free(
        self,
        contains: Optional[str] = None,
        limit: int = 20,
    ) -> List[AvailableNumber]:
        """Search for toll-free numbers."""
        return await self.search_available(
            type=PhoneNumberType.TOLL_FREE,
            contains=contains,
            limit=limit,
        )

    async def purchase_and_assign(
        self,
        phone_number: str,
        agent_id: str,
        friendly_name: Optional[str] = None,
    ) -> PhoneNumber:
        """Purchase a number and assign it to an agent in one call."""
        return await self.purchase(
            phone_number=phone_number,
            friendly_name=friendly_name,
            agent_id=agent_id,
        )


class PhoneNumberSearchBuilder:
    """
    Builder for phone number searches.

    Example:
        results = await (PhoneNumberSearchBuilder()
            .country("US")
            .type(PhoneNumberType.LOCAL)
            .area_code("415")
            .with_voice()
            .limit(10)
            .search(api))
    """

    def __init__(self):
        self._country_code: str = "US"
        self._type: PhoneNumberType = PhoneNumberType.LOCAL
        self._area_code: Optional[str] = None
        self._contains: Optional[str] = None
        self._capabilities: List[PhoneNumberCapability] = []
        self._limit: int = 20

    def country(self, code: str) -> "PhoneNumberSearchBuilder":
        """Set the country code."""
        self._country_code = code
        return self

    def type(self, number_type: PhoneNumberType) -> "PhoneNumberSearchBuilder":
        """Set the number type."""
        self._type = number_type
        return self

    def local(self) -> "PhoneNumberSearchBuilder":
        """Search for local numbers."""
        self._type = PhoneNumberType.LOCAL
        return self

    def toll_free(self) -> "PhoneNumberSearchBuilder":
        """Search for toll-free numbers."""
        self._type = PhoneNumberType.TOLL_FREE
        return self

    def mobile(self) -> "PhoneNumberSearchBuilder":
        """Search for mobile numbers."""
        self._type = PhoneNumberType.MOBILE
        return self

    def area_code(self, code: str) -> "PhoneNumberSearchBuilder":
        """Filter by area code."""
        self._area_code = code
        return self

    def contains(self, pattern: str) -> "PhoneNumberSearchBuilder":
        """Filter by pattern in number."""
        self._contains = pattern
        return self

    def with_voice(self) -> "PhoneNumberSearchBuilder":
        """Require voice capability."""
        if PhoneNumberCapability.VOICE not in self._capabilities:
            self._capabilities.append(PhoneNumberCapability.VOICE)
        return self

    def with_sms(self) -> "PhoneNumberSearchBuilder":
        """Require SMS capability."""
        if PhoneNumberCapability.SMS not in self._capabilities:
            self._capabilities.append(PhoneNumberCapability.SMS)
        return self

    def with_mms(self) -> "PhoneNumberSearchBuilder":
        """Require MMS capability."""
        if PhoneNumberCapability.MMS not in self._capabilities:
            self._capabilities.append(PhoneNumberCapability.MMS)
        return self

    def limit(self, count: int) -> "PhoneNumberSearchBuilder":
        """Set the result limit."""
        self._limit = count
        return self

    async def search(self, api: PhoneNumbersAPI) -> List[AvailableNumber]:
        """Execute the search."""
        return await api.search_available(
            country_code=self._country_code,
            type=self._type,
            area_code=self._area_code,
            contains=self._contains,
            capabilities=self._capabilities if self._capabilities else None,
            limit=self._limit,
        )

"""
Builder Voice AI SDK Client

Synchronous and asynchronous clients for the BVRAI API.
"""

import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import httpx

from .exceptions import (
    APIError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from .types import (
    Agent,
    Call,
    Conversation,
    VoiceConfiguration,
    Webhook,
    AnalyticsSummary,
    PaginatedResponse,
)


DEFAULT_BASE_URL = "https://api.buildervoiceai.com/v1"
DEFAULT_TIMEOUT = 30.0


class BaseClient:
    """Base client with common functionality."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "bvrai-python/1.0.0",
        }

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle API response and raise appropriate exceptions."""
        if response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        if response.status_code == 404:
            raise NotFoundError("Resource not found")
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            raise RateLimitError(f"Rate limit exceeded. Retry after {retry_after}s")
        if response.status_code == 422:
            raise ValidationError(response.json().get("message", "Validation error"))
        if response.status_code >= 400:
            error_data = response.json() if response.content else {}
            raise APIError(
                error_data.get("message", f"API error: {response.status_code}"),
                status_code=response.status_code,
            )
        if response.status_code == 204:
            return {}
        return response.json()


class Client(BaseClient):
    """
    Synchronous client for the Builder Voice AI API.

    Example:
        client = Client(api_key="your_api_key")
        agents = client.agents.list()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = httpx.Client(timeout=self.timeout)
        self.agents = AgentsResource(self)
        self.calls = CallsResource(self)
        self.voice_configs = VoiceConfigsResource(self)
        self.webhooks = WebhooksResource(self)
        self.analytics = AnalyticsResource(self)

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make a synchronous HTTP request."""
        url = urljoin(self.base_url + "/", path.lstrip("/"))

        for attempt in range(self.max_retries):
            try:
                response = self._client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=self._get_headers(),
                )
                return self._handle_response(response)
            except RateLimitError:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            except httpx.TransportError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise APIError(f"Network error: {e}")


class AsyncClient(BaseClient):
    """
    Asynchronous client for the Builder Voice AI API.

    Example:
        async with AsyncClient(api_key="your_api_key") as client:
            agents = await client.agents.list()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = httpx.AsyncClient(timeout=self.timeout)
        self.agents = AsyncAgentsResource(self)
        self.calls = AsyncCallsResource(self)
        self.voice_configs = AsyncVoiceConfigsResource(self)
        self.webhooks = AsyncWebhooksResource(self)
        self.analytics = AsyncAnalyticsResource(self)

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an asynchronous HTTP request."""
        import asyncio

        url = urljoin(self.base_url + "/", path.lstrip("/"))

        for attempt in range(self.max_retries):
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=self._get_headers(),
                )
                return self._handle_response(response)
            except RateLimitError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
            except httpx.TransportError as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise APIError(f"Network error: {e}")


# =============================================================================
# Synchronous Resources
# =============================================================================


class AgentsResource:
    """Agents API resource."""

    def __init__(self, client: Client):
        self._client = client

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> PaginatedResponse[Agent]:
        """List all agents."""
        params = {"page": page, "page_size": page_size}
        if status:
            params["status"] = status
        if search:
            params["search"] = search

        data = self._client._request("GET", "/agents", params=params)
        return PaginatedResponse(
            items=[Agent(**item) for item in data["items"]],
            total=data["total"],
            page=data["page"],
            page_size=data["page_size"],
        )

    def get(self, agent_id: str) -> Agent:
        """Get an agent by ID."""
        data = self._client._request("GET", f"/agents/{agent_id}")
        return Agent(**data)

    def create(
        self,
        name: str,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        voice_config_id: Optional[str] = None,
        llm_config: Optional[Dict] = None,
    ) -> Agent:
        """Create a new agent."""
        payload = {"name": name}
        if description:
            payload["description"] = description
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if voice_config_id:
            payload["voice_config_id"] = voice_config_id
        if llm_config:
            payload["llm_config"] = llm_config

        data = self._client._request("POST", "/agents", json=payload)
        return Agent(**data)

    def update(self, agent_id: str, **kwargs) -> Agent:
        """Update an agent."""
        data = self._client._request("PUT", f"/agents/{agent_id}", json=kwargs)
        return Agent(**data)

    def delete(self, agent_id: str) -> None:
        """Delete an agent."""
        self._client._request("DELETE", f"/agents/{agent_id}")

    def test(self, agent_id: str, message: str) -> Dict[str, Any]:
        """Test an agent with a message."""
        return self._client._request(
            "POST", f"/agents/{agent_id}/test", json={"message": message}
        )


class CallsResource:
    """Calls API resource."""

    def __init__(self, client: Client):
        self._client = client

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> PaginatedResponse[Call]:
        """List calls."""
        params = {"page": page, "page_size": page_size}
        if agent_id:
            params["agent_id"] = agent_id
        if status:
            params["status"] = status
        if direction:
            params["direction"] = direction

        data = self._client._request("GET", "/calls", params=params)
        return PaginatedResponse(
            items=[Call(**item) for item in data["items"]],
            total=data["total"],
            page=data["page"],
            page_size=data["page_size"],
        )

    def get(self, call_id: str) -> Call:
        """Get a call by ID."""
        data = self._client._request("GET", f"/calls/{call_id}")
        return Call(**data)

    def get_conversation(self, call_id: str) -> Conversation:
        """Get conversation for a call."""
        data = self._client._request("GET", f"/calls/{call_id}/conversation")
        return Conversation(**data)

    def initiate_outbound(
        self,
        agent_id: str,
        to_number: str,
        metadata: Optional[Dict] = None,
    ) -> Call:
        """Initiate an outbound call."""
        payload = {"agent_id": agent_id, "to_number": to_number}
        if metadata:
            payload["metadata"] = metadata

        data = self._client._request("POST", "/calls/outbound", json=payload)
        return Call(**data)

    def hangup(self, call_id: str) -> None:
        """Hangup a call."""
        self._client._request("POST", f"/calls/{call_id}/hangup")


class VoiceConfigsResource:
    """Voice configurations API resource."""

    def __init__(self, client: Client):
        self._client = client

    def list(self) -> List[VoiceConfiguration]:
        """List voice configurations."""
        data = self._client._request("GET", "/voice-configs")
        return [VoiceConfiguration(**item) for item in data]

    def get(self, config_id: str) -> VoiceConfiguration:
        """Get a voice configuration."""
        data = self._client._request("GET", f"/voice-configs/{config_id}")
        return VoiceConfiguration(**data)

    def create(self, **kwargs) -> VoiceConfiguration:
        """Create a voice configuration."""
        data = self._client._request("POST", "/voice-configs", json=kwargs)
        return VoiceConfiguration(**data)


class WebhooksResource:
    """Webhooks API resource."""

    def __init__(self, client: Client):
        self._client = client

    def list(self) -> List[Webhook]:
        """List webhooks."""
        data = self._client._request("GET", "/webhooks")
        return [Webhook(**item) for item in data]

    def create(
        self,
        url: str,
        events: List[str],
        name: Optional[str] = None,
    ) -> Webhook:
        """Create a webhook."""
        payload = {"url": url, "events": events}
        if name:
            payload["name"] = name

        data = self._client._request("POST", "/webhooks", json=payload)
        return Webhook(**data)

    def delete(self, webhook_id: str) -> None:
        """Delete a webhook."""
        self._client._request("DELETE", f"/webhooks/{webhook_id}")

    def test(self, webhook_id: str) -> Dict[str, Any]:
        """Test a webhook."""
        return self._client._request("POST", f"/webhooks/{webhook_id}/test")


class AnalyticsResource:
    """Analytics API resource."""

    def __init__(self, client: Client):
        self._client = client

    def get_summary(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> AnalyticsSummary:
        """Get analytics summary."""
        params = {}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
        if agent_id:
            params["agent_id"] = agent_id

        data = self._client._request("GET", "/analytics/summary", params=params)
        return AnalyticsSummary(**data)

    def get_dashboard(self) -> Dict[str, Any]:
        """Get dashboard stats."""
        return self._client._request("GET", "/analytics/dashboard")


# =============================================================================
# Async Resources (mirrors sync resources)
# =============================================================================


class AsyncAgentsResource:
    """Async Agents API resource."""

    def __init__(self, client: AsyncClient):
        self._client = client

    async def list(self, **kwargs) -> PaginatedResponse[Agent]:
        params = {k: v for k, v in kwargs.items() if v is not None}
        data = await self._client._request("GET", "/agents", params=params)
        return PaginatedResponse(
            items=[Agent(**item) for item in data["items"]],
            total=data["total"],
            page=data["page"],
            page_size=data["page_size"],
        )

    async def get(self, agent_id: str) -> Agent:
        data = await self._client._request("GET", f"/agents/{agent_id}")
        return Agent(**data)

    async def create(self, **kwargs) -> Agent:
        data = await self._client._request("POST", "/agents", json=kwargs)
        return Agent(**data)

    async def update(self, agent_id: str, **kwargs) -> Agent:
        data = await self._client._request("PUT", f"/agents/{agent_id}", json=kwargs)
        return Agent(**data)

    async def delete(self, agent_id: str) -> None:
        await self._client._request("DELETE", f"/agents/{agent_id}")


class AsyncCallsResource:
    """Async Calls API resource."""

    def __init__(self, client: AsyncClient):
        self._client = client

    async def list(self, **kwargs) -> PaginatedResponse[Call]:
        params = {k: v for k, v in kwargs.items() if v is not None}
        data = await self._client._request("GET", "/calls", params=params)
        return PaginatedResponse(
            items=[Call(**item) for item in data["items"]],
            total=data["total"],
            page=data["page"],
            page_size=data["page_size"],
        )

    async def get(self, call_id: str) -> Call:
        data = await self._client._request("GET", f"/calls/{call_id}")
        return Call(**data)


class AsyncVoiceConfigsResource:
    """Async Voice configs resource."""

    def __init__(self, client: AsyncClient):
        self._client = client

    async def list(self) -> List[VoiceConfiguration]:
        data = await self._client._request("GET", "/voice-configs")
        return [VoiceConfiguration(**item) for item in data]


class AsyncWebhooksResource:
    """Async Webhooks resource."""

    def __init__(self, client: AsyncClient):
        self._client = client

    async def list(self) -> List[Webhook]:
        data = await self._client._request("GET", "/webhooks")
        return [Webhook(**item) for item in data]


class AsyncAnalyticsResource:
    """Async Analytics resource."""

    def __init__(self, client: AsyncClient):
        self._client = client

    async def get_summary(self, **kwargs) -> AnalyticsSummary:
        params = {k: v for k, v in kwargs.items() if v is not None}
        data = await self._client._request("GET", "/analytics/summary", params=params)
        return AnalyticsSummary(**data)

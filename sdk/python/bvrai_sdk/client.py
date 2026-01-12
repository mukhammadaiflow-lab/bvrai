"""Main client for Builder Engine Voice AI API."""

import asyncio
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from bvrai_sdk.agents import AgentsAPI
from bvrai_sdk.calls import CallsAPI
from bvrai_sdk.knowledge import KnowledgeAPI
from bvrai_sdk.analytics import AnalyticsAPI
from bvrai_sdk.webhooks import WebhooksAPI
from bvrai_sdk.phone_numbers import PhoneNumbersAPI
from bvrai_sdk.exceptions import (
    BvraiError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ValidationError,
    ServerError,
    ConnectionError as BvraiConnectionError,
)


@dataclass
class ClientConfig:
    """Client configuration."""
    api_key: str
    base_url: str = "https://api.bvrai.com"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class BvraiClient:
    """
    Builder Engine Voice AI API Client.

    Usage:
        client = BvraiClient(api_key="your-api-key")

        # Create an agent
        agent = await client.agents.create(
            name="Sales Assistant",
            system_prompt="You are a helpful sales assistant...",
        )

        # Make an outbound call
        call = await client.calls.create(
            agent_id=agent.id,
            to_number="+15551234567",
        )

        # Check call status
        status = await client.calls.get(call.id)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.bvrai.com",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize the client.

        Args:
            api_key: Your API key
            base_url: API base URL (default: production)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.config = ClientConfig(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            max_retries=max_retries,
        )

        self._http_client: Optional[httpx.AsyncClient] = None

        # Initialize API modules
        self.agents = AgentsAPI(self)
        self.calls = CallsAPI(self)
        self.knowledge = KnowledgeAPI(self)
        self.analytics = AnalyticsAPI(self)
        self.webhooks = WebhooksAPI(self)
        self.phone_numbers = PhoneNumbersAPI(self)

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "bvrai-python-sdk/1.0.0",
                },
            )
        return self._http_client

    async def close(self) -> None:
        """Close the client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an API request.

        Args:
            method: HTTP method
            path: API path
            data: Request body
            params: Query parameters

        Returns:
            Response data

        Raises:
            BvraiError: On API errors
        """
        client = await self._ensure_client()

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await client.request(
                    method=method,
                    url=path,
                    json=data,
                    params=params,
                )

                # Handle errors
                if response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status_code == 404:
                    raise NotFoundError(f"Resource not found: {path}")
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    raise RateLimitError(
                        f"Rate limit exceeded. Retry after {retry_after}s",
                        retry_after=retry_after,
                    )
                elif response.status_code == 422:
                    error_data = response.json()
                    raise ValidationError(
                        error_data.get("detail", "Validation error"),
                        errors=error_data.get("errors", []),
                    )
                elif response.status_code >= 400:
                    error_data = response.json()
                    raise BvraiError(
                        error_data.get("detail", "API error"),
                        status_code=response.status_code,
                    )

                # Return successful response
                if response.status_code == 204:
                    return {}

                return response.json()

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                continue

        raise BvraiError(f"Request failed after {self.config.max_retries} retries: {last_error}")

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make GET request."""
        return await self.request("GET", path, params=params)

    async def post(self, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make POST request."""
        return await self.request("POST", path, data=data)

    async def put(self, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make PUT request."""
        return await self.request("PUT", path, data=data)

    async def patch(self, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make PATCH request."""
        return await self.request("PATCH", path, data=data)

    async def delete(self, path: str) -> Dict[str, Any]:
        """Make DELETE request."""
        return await self.request("DELETE", path)

    async def get_raw(self, path: str, params: Optional[Dict[str, Any]] = None) -> bytes:
        """Make GET request and return raw bytes."""
        client = await self._ensure_client()

        response = await client.request(
            method="GET",
            url=path,
            params=params,
        )

        if response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif response.status_code == 404:
            raise NotFoundError(f"Resource not found: {path}")
        elif response.status_code >= 400:
            raise BvraiError(f"Request failed: {response.status_code}")

        return response.content

    async def health_check(self) -> bool:
        """Check API health."""
        try:
            response = await self.get("/health")
            return response.get("status") == "healthy"
        except Exception:
            return False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._http_client is not None


# Synchronous wrapper for non-async usage
class SyncBvraiClient:
    """
    Synchronous wrapper for BvraiClient.

    For use in non-async contexts.
    """

    def __init__(self, api_key: str, **kwargs):
        self._async_client = BvraiClient(api_key, **kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
            return self._loop

    def _run(self, coro):
        """Run coroutine synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def close(self) -> None:
        """Close the client."""
        self._run(self._async_client.close())
        if self._loop:
            self._loop.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # Proxy to async client methods
    @property
    def agents(self):
        return SyncAPIWrapper(self._async_client.agents, self._run)

    @property
    def calls(self):
        return SyncAPIWrapper(self._async_client.calls, self._run)


class SyncAPIWrapper:
    """Wrapper to make async API calls synchronous."""

    def __init__(self, async_api, runner):
        self._async_api = async_api
        self._run = runner

    def __getattr__(self, name):
        attr = getattr(self._async_api, name)
        if callable(attr):
            def sync_wrapper(*args, **kwargs):
                return self._run(attr(*args, **kwargs))
            return sync_wrapper
        return attr

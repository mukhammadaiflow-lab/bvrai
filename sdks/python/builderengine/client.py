"""
Builder Engine Python SDK - Main Client

This module provides the main BuilderEngine client class that serves as
the entry point for all API interactions.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any, Union
from urllib.parse import urljoin

import httpx

from builderengine.config import ClientConfig, DEFAULT_CONFIG
from builderengine.exceptions import (
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError,
    ConflictError,
    ServerError,
    BuilderEngineError,
)
from builderengine.resources.agents import AgentsResource
from builderengine.resources.calls import CallsResource
from builderengine.resources.conversations import ConversationsResource
from builderengine.resources.phone_numbers import PhoneNumbersResource
from builderengine.resources.voices import VoicesResource
from builderengine.resources.webhooks import WebhooksResource
from builderengine.resources.knowledge_base import KnowledgeBaseResource
from builderengine.resources.workflows import WorkflowsResource
from builderengine.resources.campaigns import CampaignsResource
from builderengine.resources.analytics import AnalyticsResource
from builderengine.resources.organizations import OrganizationsResource
from builderengine.resources.users import UsersResource
from builderengine.resources.api_keys import APIKeysResource
from builderengine.resources.billing import BillingResource
from builderengine.streaming import StreamingClient

logger = logging.getLogger("builderengine")


class BuilderEngine:
    """
    Main client for interacting with the Builder Engine API.

    This client provides access to all Builder Engine resources including
    agents, calls, conversations, phone numbers, and more.

    Args:
        api_key: Your Builder Engine API key. If not provided, will look for
            BUILDERENGINE_API_KEY environment variable.
        base_url: The base URL for the API. Defaults to https://api.builderengine.io
        timeout: Request timeout in seconds. Defaults to 30.
        max_retries: Maximum number of retry attempts. Defaults to 3.
        organization_id: Organization ID for multi-tenant requests.
        debug: Enable debug logging. Defaults to False.

    Example:
        >>> client = BuilderEngine(api_key="your-api-key")
        >>> agents = client.agents.list()
        >>> for agent in agents:
        ...     print(agent.name)

    Attributes:
        agents: Resource for managing AI agents
        calls: Resource for managing voice calls
        conversations: Resource for managing conversations
        phone_numbers: Resource for managing phone numbers
        voices: Resource for managing voice configurations
        webhooks: Resource for managing webhooks
        knowledge_base: Resource for managing knowledge bases
        workflows: Resource for managing workflows
        campaigns: Resource for managing call campaigns
        analytics: Resource for accessing analytics data
        organizations: Resource for managing organizations
        users: Resource for managing users
        api_keys: Resource for managing API keys
        billing: Resource for billing and usage
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        organization_id: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        # Get API key from parameter or environment
        self._api_key = api_key or os.environ.get("BUILDERENGINE_API_KEY")
        if not self._api_key:
            raise AuthenticationError(
                "API key is required. Provide it as a parameter or set "
                "the BUILDERENGINE_API_KEY environment variable."
            )

        # Configuration
        self._config = ClientConfig(
            base_url=base_url or os.environ.get(
                "BUILDERENGINE_BASE_URL",
                DEFAULT_CONFIG.base_url
            ),
            timeout=timeout,
            max_retries=max_retries,
            organization_id=organization_id,
            debug=debug,
        )

        # Setup logging
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Create HTTP client
        self._http_client = self._create_http_client()

        # Initialize resources
        self._init_resources()

        logger.debug(f"BuilderEngine client initialized with base URL: {self._config.base_url}")

    def _create_http_client(self) -> httpx.Client:
        """Create and configure the HTTP client."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"builderengine-python/1.0.0",
        }

        if self._config.organization_id:
            headers["X-Organization-ID"] = self._config.organization_id

        transport = httpx.HTTPTransport(retries=self._config.max_retries)

        return httpx.Client(
            base_url=self._config.base_url,
            headers=headers,
            timeout=httpx.Timeout(self._config.timeout),
            transport=transport,
            follow_redirects=True,
        )

    def _init_resources(self) -> None:
        """Initialize all API resources."""
        self.agents = AgentsResource(self)
        self.calls = CallsResource(self)
        self.conversations = ConversationsResource(self)
        self.phone_numbers = PhoneNumbersResource(self)
        self.voices = VoicesResource(self)
        self.webhooks = WebhooksResource(self)
        self.knowledge_base = KnowledgeBaseResource(self)
        self.workflows = WorkflowsResource(self)
        self.campaigns = CampaignsResource(self)
        self.analytics = AnalyticsResource(self)
        self.organizations = OrganizationsResource(self)
        self.users = UsersResource(self)
        self.api_keys = APIKeysResource(self)
        self.billing = BillingResource(self)

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            path: API endpoint path
            params: Query parameters
            json: JSON body data
            data: Form data
            files: Files to upload
            headers: Additional headers

        Returns:
            Response data as dictionary

        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            ValidationError: If request validation fails
            NotFoundError: If resource is not found
            ConflictError: If there's a resource conflict
            ServerError: If server error occurs
            BuilderEngineError: For other errors
        """
        url = path if path.startswith("http") else path

        request_headers = dict(self._http_client.headers)
        if headers:
            request_headers.update(headers)

        # Remove Content-Type for file uploads
        if files:
            request_headers.pop("Content-Type", None)

        logger.debug(f"Making {method} request to {url}")
        logger.debug(f"Params: {params}")
        logger.debug(f"JSON: {json}")

        try:
            response = self._http_client.request(
                method=method,
                url=url,
                params=params,
                json=json,
                data=data,
                files=files,
                headers=request_headers,
            )

            return self._handle_response(response)

        except httpx.TimeoutException as e:
            raise BuilderEngineError(f"Request timed out: {e}")
        except httpx.RequestError as e:
            raise BuilderEngineError(f"Request failed: {e}")

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle API response and raise appropriate exceptions."""
        logger.debug(f"Response status: {response.status_code}")

        # Success responses
        if response.status_code in (200, 201, 202):
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.json()
            return {"data": response.text}

        # No content
        if response.status_code == 204:
            return {}

        # Error responses
        try:
            error_data = response.json()
            error_message = error_data.get("detail") or error_data.get("message") or str(error_data)
        except Exception:
            error_message = response.text or f"HTTP {response.status_code}"

        if response.status_code == 401:
            raise AuthenticationError(error_message)
        elif response.status_code == 403:
            raise AuthenticationError(f"Forbidden: {error_message}")
        elif response.status_code == 404:
            raise NotFoundError(error_message)
        elif response.status_code == 409:
            raise ConflictError(error_message)
        elif response.status_code == 422:
            raise ValidationError(error_message)
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(error_message, retry_after=retry_after)
        elif response.status_code >= 500:
            raise ServerError(error_message)
        else:
            raise BuilderEngineError(f"HTTP {response.status_code}: {error_message}")

    def streaming(self) -> StreamingClient:
        """
        Get a streaming client for real-time events.

        Returns:
            StreamingClient for WebSocket connections

        Example:
            >>> streaming = client.streaming()
            >>> async with streaming.connect() as ws:
            ...     async for event in ws.events():
            ...         print(event)
        """
        return StreamingClient(
            api_key=self._api_key,
            base_url=self._config.base_url.replace("https://", "wss://").replace("http://", "ws://"),
        )

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        self._http_client.close()
        logger.debug("BuilderEngine client closed")

    def __enter__(self) -> "BuilderEngine":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        return f"BuilderEngine(base_url='{self._config.base_url}')"


class AsyncBuilderEngine:
    """
    Async client for interacting with the Builder Engine API.

    This client provides async/await support for all Builder Engine resources.

    Args:
        api_key: Your Builder Engine API key.
        base_url: The base URL for the API.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        organization_id: Organization ID for multi-tenant requests.
        debug: Enable debug logging.

    Example:
        >>> async with AsyncBuilderEngine(api_key="your-api-key") as client:
        ...     agents = await client.agents.list()
        ...     for agent in agents:
        ...         print(agent.name)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        organization_id: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        self._api_key = api_key or os.environ.get("BUILDERENGINE_API_KEY")
        if not self._api_key:
            raise AuthenticationError(
                "API key is required. Provide it as a parameter or set "
                "the BUILDERENGINE_API_KEY environment variable."
            )

        self._config = ClientConfig(
            base_url=base_url or os.environ.get(
                "BUILDERENGINE_BASE_URL",
                DEFAULT_CONFIG.base_url
            ),
            timeout=timeout,
            max_retries=max_retries,
            organization_id=organization_id,
            debug=debug,
        )

        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        self._http_client: Optional[httpx.AsyncClient] = None
        self._init_resources()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._http_client is None:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": f"builderengine-python/1.0.0",
            }

            if self._config.organization_id:
                headers["X-Organization-ID"] = self._config.organization_id

            transport = httpx.AsyncHTTPTransport(retries=self._config.max_retries)

            self._http_client = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers=headers,
                timeout=httpx.Timeout(self._config.timeout),
                transport=transport,
                follow_redirects=True,
            )

        return self._http_client

    def _init_resources(self) -> None:
        """Initialize all API resources with async support."""
        from builderengine.resources.async_resources import (
            AsyncAgentsResource,
            AsyncCallsResource,
            AsyncConversationsResource,
            AsyncPhoneNumbersResource,
            AsyncVoicesResource,
            AsyncWebhooksResource,
            AsyncKnowledgeBaseResource,
            AsyncWorkflowsResource,
            AsyncCampaignsResource,
            AsyncAnalyticsResource,
            AsyncOrganizationsResource,
            AsyncUsersResource,
            AsyncAPIKeysResource,
            AsyncBillingResource,
        )

        self.agents = AsyncAgentsResource(self)
        self.calls = AsyncCallsResource(self)
        self.conversations = AsyncConversationsResource(self)
        self.phone_numbers = AsyncPhoneNumbersResource(self)
        self.voices = AsyncVoicesResource(self)
        self.webhooks = AsyncWebhooksResource(self)
        self.knowledge_base = AsyncKnowledgeBaseResource(self)
        self.workflows = AsyncWorkflowsResource(self)
        self.campaigns = AsyncCampaignsResource(self)
        self.analytics = AsyncAnalyticsResource(self)
        self.organizations = AsyncOrganizationsResource(self)
        self.users = AsyncUsersResource(self)
        self.api_keys = AsyncAPIKeysResource(self)
        self.billing = AsyncBillingResource(self)

    async def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make an async HTTP request to the API."""
        client = await self._get_client()

        request_headers = dict(client.headers)
        if headers:
            request_headers.update(headers)

        if files:
            request_headers.pop("Content-Type", None)

        logger.debug(f"Making async {method} request to {path}")

        try:
            response = await client.request(
                method=method,
                url=path,
                params=params,
                json=json,
                data=data,
                files=files,
                headers=request_headers,
            )

            return self._handle_response(response)

        except httpx.TimeoutException as e:
            raise BuilderEngineError(f"Request timed out: {e}")
        except httpx.RequestError as e:
            raise BuilderEngineError(f"Request failed: {e}")

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle API response and raise appropriate exceptions."""
        if response.status_code in (200, 201, 202):
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.json()
            return {"data": response.text}

        if response.status_code == 204:
            return {}

        try:
            error_data = response.json()
            error_message = error_data.get("detail") or error_data.get("message") or str(error_data)
        except Exception:
            error_message = response.text or f"HTTP {response.status_code}"

        if response.status_code == 401:
            raise AuthenticationError(error_message)
        elif response.status_code == 403:
            raise AuthenticationError(f"Forbidden: {error_message}")
        elif response.status_code == 404:
            raise NotFoundError(error_message)
        elif response.status_code == 409:
            raise ConflictError(error_message)
        elif response.status_code == 422:
            raise ValidationError(error_message)
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(error_message, retry_after=retry_after)
        elif response.status_code >= 500:
            raise ServerError(error_message)
        else:
            raise BuilderEngineError(f"HTTP {response.status_code}: {error_message}")

    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug("AsyncBuilderEngine client closed")

    async def __aenter__(self) -> "AsyncBuilderEngine":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        return f"AsyncBuilderEngine(base_url='{self._config.base_url}')"

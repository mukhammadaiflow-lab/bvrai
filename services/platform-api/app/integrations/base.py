"""Base classes for integrations."""

from typing import Optional, Dict, Any, List, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging

logger = logging.getLogger(__name__)


class IntegrationStatus(str, Enum):
    """Integration connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    EXPIRED = "expired"


@dataclass
class IntegrationConfig:
    """Configuration for an integration."""
    integration_id: str
    user_id: str
    provider: str
    credentials: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    status: IntegrationStatus = IntegrationStatus.DISCONNECTED
    connected_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without sensitive data)."""
        return {
            "integration_id": self.integration_id,
            "user_id": self.user_id,
            "provider": self.provider,
            "settings": self.settings,
            "status": self.status.value,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class Integration(ABC):
    """
    Base class for all integrations.

    Provides common functionality for OAuth, API calls, and error handling.
    """

    PROVIDER_NAME: str = "base"
    REQUIRED_SCOPES: List[str] = []

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self._http_client = None

    @property
    def is_connected(self) -> bool:
        """Check if integration is connected."""
        return self.config.status == IntegrationStatus.CONNECTED

    @abstractmethod
    async def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect the integration."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect the integration."""
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the connection is valid."""
        pass

    @abstractmethod
    async def refresh_token(self) -> bool:
        """Refresh OAuth token if applicable."""
        pass

    async def _get_http_client(self):
        """Get HTTP client."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make an authenticated API request."""
        client = await self._get_http_client()

        # Add authorization header
        headers = kwargs.pop("headers", {})
        access_token = self.config.credentials.get("access_token")
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        kwargs["headers"] = headers

        response = await client.request(method, url, **kwargs)

        if response.status_code == 401:
            # Try to refresh token
            refreshed = await self.refresh_token()
            if refreshed:
                headers["Authorization"] = f"Bearer {self.config.credentials.get('access_token')}"
                response = await client.request(method, url, **kwargs)

        response.raise_for_status()
        return response.json()

    def get_oauth_url(self, redirect_uri: str, state: str) -> str:
        """Get OAuth authorization URL."""
        raise NotImplementedError("OAuth not supported")

    async def exchange_code(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange OAuth code for tokens."""
        raise NotImplementedError("OAuth not supported")


class IntegrationRegistry:
    """
    Registry for managing integration instances.

    Usage:
        registry = IntegrationRegistry()
        registry.register_provider("salesforce", SalesforceIntegration)

        # Get or create integration
        integration = await registry.get_integration(user_id, "salesforce")
    """

    def __init__(self):
        self._providers: Dict[str, Type[Integration]] = {}
        self._instances: Dict[str, Integration] = {}
        self._configs: Dict[str, IntegrationConfig] = {}

    def register_provider(
        self,
        name: str,
        integration_class: Type[Integration],
    ) -> None:
        """Register an integration provider."""
        self._providers[name] = integration_class
        logger.info(f"Registered integration provider: {name}")

    def get_provider(self, name: str) -> Optional[Type[Integration]]:
        """Get an integration provider class."""
        return self._providers.get(name)

    def list_providers(self) -> List[str]:
        """List all registered providers."""
        return list(self._providers.keys())

    async def create_integration(
        self,
        user_id: str,
        provider: str,
        credentials: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Integration:
        """Create a new integration instance."""
        provider_class = self._providers.get(provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider}")

        import uuid
        config = IntegrationConfig(
            integration_id=str(uuid.uuid4()),
            user_id=user_id,
            provider=provider,
            credentials=credentials or {},
            settings=settings or {},
        )

        integration = provider_class(config)
        self._configs[config.integration_id] = config
        self._instances[config.integration_id] = integration

        return integration

    async def get_integration(
        self,
        user_id: str,
        provider: str,
    ) -> Optional[Integration]:
        """Get existing integration for user and provider."""
        for config in self._configs.values():
            if config.user_id == user_id and config.provider == provider:
                return self._instances.get(config.integration_id)
        return None

    async def get_user_integrations(self, user_id: str) -> List[Integration]:
        """Get all integrations for a user."""
        integrations = []
        for config in self._configs.values():
            if config.user_id == user_id:
                integration = self._instances.get(config.integration_id)
                if integration:
                    integrations.append(integration)
        return integrations

    async def remove_integration(self, integration_id: str) -> bool:
        """Remove an integration."""
        if integration_id in self._instances:
            integration = self._instances.pop(integration_id)
            await integration.disconnect()
            self._configs.pop(integration_id, None)
            return True
        return False


# Global registry
_integration_registry: Optional[IntegrationRegistry] = None


def get_integration_registry() -> IntegrationRegistry:
    """Get or create the global integration registry."""
    global _integration_registry
    if _integration_registry is None:
        _integration_registry = IntegrationRegistry()
    return _integration_registry

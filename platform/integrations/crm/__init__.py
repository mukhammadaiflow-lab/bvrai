"""
CRM Integrations Package

This package provides CRM provider implementations for the voice agent platform,
enabling contact management, call logging, and deal tracking across popular
CRM platforms.

Supported Providers:
- Salesforce: Enterprise CRM with full API access
- HubSpot: Marketing and sales platform with free tier

Example usage:

    from platform.integrations.crm import (
        SalesforceProvider,
        HubSpotProvider,
        get_crm_provider,
    )

    # Create Salesforce provider
    salesforce = SalesforceProvider(
        integration_id="int_123",
        organization_id="org_456",
        credentials={
            "access_token": "...",
            "refresh_token": "...",
            "instance_url": "https://company.salesforce.com",
        },
    )

    # List contacts
    contacts, next_cursor = await salesforce.list_contacts(limit=100)

    # Log a call
    call_log_id = await salesforce.log_call(CallLogEntry(...))

    # Or use factory function
    provider = get_crm_provider(
        provider_name="hubspot",
        integration_id="int_789",
        organization_id="org_456",
        credentials={"access_token": "..."},
    )
"""

from typing import Dict, Any, Optional, Type

from ..base import CRMProvider, IntegrationError
from .salesforce import SalesforceProvider
from .hubspot import HubSpotProvider


# Registry of available CRM providers
CRM_PROVIDERS: Dict[str, Type[CRMProvider]] = {
    "salesforce": SalesforceProvider,
    "hubspot": HubSpotProvider,
}


def get_crm_provider(
    provider_name: str,
    integration_id: str,
    organization_id: str,
    credentials: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
) -> CRMProvider:
    """
    Factory function to create a CRM provider instance.

    Args:
        provider_name: Name of the provider (salesforce, hubspot)
        integration_id: Unique integration identifier
        organization_id: Organization identifier
        credentials: Provider-specific credentials
        settings: Optional provider settings

    Returns:
        Configured CRM provider instance

    Raises:
        IntegrationError: If provider is not supported
    """
    provider_class = CRM_PROVIDERS.get(provider_name.lower())
    if not provider_class:
        raise IntegrationError(
            f"Unsupported CRM provider: {provider_name}. "
            f"Available providers: {', '.join(CRM_PROVIDERS.keys())}"
        )

    return provider_class(
        integration_id=integration_id,
        organization_id=organization_id,
        credentials=credentials,
        settings=settings or {},
    )


def list_crm_providers() -> Dict[str, Dict[str, Any]]:
    """
    List all available CRM providers with their metadata.

    Returns:
        Dictionary of provider information
    """
    providers = {}
    for name, provider_class in CRM_PROVIDERS.items():
        providers[name] = {
            "name": provider_class.PROVIDER_NAME,
            "auth_type": provider_class.AUTH_TYPE.value,
            "capabilities": [
                "list_contacts",
                "get_contact",
                "create_contact",
                "update_contact",
                "search_contacts",
                "find_by_phone",
                "log_call",
                "list_deals",
                "get_deal",
            ],
        }
    return providers


__all__ = [
    # Providers
    "SalesforceProvider",
    "HubSpotProvider",
    # Factory
    "get_crm_provider",
    "list_crm_providers",
    # Registry
    "CRM_PROVIDERS",
]

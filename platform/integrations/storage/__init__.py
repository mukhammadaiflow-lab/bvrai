"""
Storage Integrations Package

This package provides storage provider implementations for the voice agent platform,
enabling file storage, retrieval, and management for call recordings, transcripts,
and other assets across cloud storage platforms.

Supported Providers:
- AWS S3: Amazon Simple Storage Service with full API support

Example usage:

    from platform.integrations.storage import (
        S3Provider,
        get_storage_provider,
    )

    # Create S3 provider
    s3 = S3Provider(
        integration_id="int_123",
        organization_id="org_456",
        credentials={
            "access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        },
        settings={
            "bucket": "my-recordings-bucket",
            "region": "us-east-1",
        },
    )

    # Upload file
    obj = await s3.upload_object(
        key="recordings/call_123.mp3",
        data=audio_bytes,
        content_type="audio/mpeg",
    )

    # Generate presigned URL for sharing
    url = s3.generate_presigned_url(
        key="recordings/call_123.mp3",
        expires_in=3600,
    )

    # Download file
    data = await s3.download_object("recordings/call_123.mp3")
"""

from typing import Dict, Any, Optional, Type

from ..base import StorageProvider, IntegrationError
from .s3 import S3Provider


# Registry of available storage providers
STORAGE_PROVIDERS: Dict[str, Type[StorageProvider]] = {
    "s3": S3Provider,
    "aws_s3": S3Provider,  # Alias
    "amazon_s3": S3Provider,  # Alias
}


def get_storage_provider(
    provider_name: str,
    integration_id: str,
    organization_id: str,
    credentials: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
) -> StorageProvider:
    """
    Factory function to create a storage provider instance.

    Args:
        provider_name: Name of the provider (s3)
        integration_id: Unique integration identifier
        organization_id: Organization identifier
        credentials: Provider-specific credentials
        settings: Optional provider settings

    Returns:
        Configured storage provider instance

    Raises:
        IntegrationError: If provider is not supported
    """
    provider_class = STORAGE_PROVIDERS.get(provider_name.lower())
    if not provider_class:
        raise IntegrationError(
            f"Unsupported storage provider: {provider_name}. "
            f"Available providers: s3"
        )

    return provider_class(
        integration_id=integration_id,
        organization_id=organization_id,
        credentials=credentials,
        settings=settings or {},
    )


def list_storage_providers() -> Dict[str, Dict[str, Any]]:
    """
    List all available storage providers with their metadata.

    Returns:
        Dictionary of provider information
    """
    # Deduplicate aliases
    unique_providers = {
        "s3": S3Provider,
    }

    providers = {}
    for name, provider_class in unique_providers.items():
        providers[name] = {
            "name": provider_class.PROVIDER_NAME,
            "auth_type": provider_class.AUTH_TYPE.value,
            "capabilities": [
                "list_objects",
                "get_object",
                "upload_object",
                "download_object",
                "delete_object",
                "copy_object",
                "generate_presigned_url",
                "multipart_upload",
                "list_buckets",
                "create_bucket",
            ],
        }
    return providers


__all__ = [
    # Providers
    "S3Provider",
    # Factory
    "get_storage_provider",
    "list_storage_providers",
    # Registry
    "STORAGE_PROVIDERS",
]

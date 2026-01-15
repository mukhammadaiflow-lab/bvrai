"""
Voice Storage Service.

Handles storage of audio samples and voice assets across
multiple storage backends (S3, GCS, Azure, Local).
"""

import asyncio
import hashlib
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from ..config import get_settings, StorageConfig

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend interface."""

    @abstractmethod
    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload data and return URL/path."""
        pass

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Download data by key."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data by key."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Get a signed URL for the key."""
        pass

    @abstractmethod
    async def list_keys(self, prefix: str) -> List[str]:
        """List keys with prefix."""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get full path for a key."""
        # Sanitize key to prevent path traversal
        safe_key = key.replace("..", "").lstrip("/")
        return self.base_path / safe_key

    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload data to local filesystem."""
        path = self._get_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write data
        await asyncio.to_thread(path.write_bytes, data)

        # Write metadata if provided
        if metadata:
            meta_path = path.with_suffix(path.suffix + ".meta")
            import json
            await asyncio.to_thread(
                meta_path.write_text,
                json.dumps(metadata),
            )

        return str(path)

    async def download(self, key: str) -> bytes:
        """Download data from local filesystem."""
        path = self._get_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Key not found: {key}")
        return await asyncio.to_thread(path.read_bytes)

    async def delete(self, key: str) -> bool:
        """Delete from local filesystem."""
        path = self._get_path(key)
        try:
            if path.exists():
                path.unlink()
            # Also delete metadata
            meta_path = path.with_suffix(path.suffix + ".meta")
            if meta_path.exists():
                meta_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self._get_path(key).exists()

    async def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Get file path (local storage doesn't have signed URLs)."""
        path = self._get_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Key not found: {key}")
        return f"file://{path}"

    async def list_keys(self, prefix: str) -> List[str]:
        """List files with prefix."""
        prefix_path = self._get_path(prefix)
        if not prefix_path.exists():
            return []

        keys = []
        for path in prefix_path.rglob("*"):
            if path.is_file() and not path.suffix == ".meta":
                rel_path = path.relative_to(self.base_path)
                keys.append(str(rel_path))
        return keys


class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self.bucket = bucket
        self.region = region
        self._client = None

        # Lazy import boto3
        try:
            import boto3
            self._s3 = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
        except ImportError:
            logger.warning("boto3 not installed, S3 backend unavailable")
            self._s3 = None

    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload to S3."""
        if not self._s3:
            raise RuntimeError("S3 client not available")

        await asyncio.to_thread(
            self._s3.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata=metadata or {},
        )

        return f"s3://{self.bucket}/{key}"

    async def download(self, key: str) -> bytes:
        """Download from S3."""
        if not self._s3:
            raise RuntimeError("S3 client not available")

        response = await asyncio.to_thread(
            self._s3.get_object,
            Bucket=self.bucket,
            Key=key,
        )
        return response["Body"].read()

    async def delete(self, key: str) -> bool:
        """Delete from S3."""
        if not self._s3:
            return False

        try:
            await asyncio.to_thread(
                self._s3.delete_object,
                Bucket=self.bucket,
                Key=key,
            )
            return True
        except Exception as e:
            logger.error(f"S3 delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in S3."""
        if not self._s3:
            return False

        try:
            await asyncio.to_thread(
                self._s3.head_object,
                Bucket=self.bucket,
                Key=key,
            )
            return True
        except Exception:
            return False

    async def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate presigned URL."""
        if not self._s3:
            raise RuntimeError("S3 client not available")

        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    async def list_keys(self, prefix: str) -> List[str]:
        """List keys with prefix."""
        if not self._s3:
            return []

        response = await asyncio.to_thread(
            self._s3.list_objects_v2,
            Bucket=self.bucket,
            Prefix=prefix,
        )
        return [obj["Key"] for obj in response.get("Contents", [])]


class VoiceStorage:
    """
    Voice storage service.

    Handles storage operations for audio samples and voice assets
    with a unified interface across storage backends.
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or get_settings().storage
        self._backend = self._create_backend()

    def _create_backend(self) -> StorageBackend:
        """Create storage backend based on configuration."""
        if self.config.provider == "s3":
            return S3StorageBackend(
                bucket=self.config.bucket_name,
                region=self.config.region,
                access_key=self.config.aws_access_key_id,
                secret_key=self.config.aws_secret_access_key,
            )
        else:  # Default to local
            return LocalStorageBackend(self.config.local_path)

    def _generate_key(
        self,
        tenant_id: str,
        category: str,
        filename: str,
    ) -> str:
        """Generate a storage key."""
        # Structure: tenant_id/category/date/filename
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ".-_")
        return f"{tenant_id}/{category}/{date_prefix}/{unique_id}_{safe_filename}"

    async def store_sample(
        self,
        tenant_id: str,
        audio_data: bytes,
        filename: str,
        content_type: str = "audio/wav",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """
        Store an audio sample.

        Returns:
            Tuple of (storage_key, storage_url)
        """
        key = self._generate_key(tenant_id, "samples", filename)

        # Add checksum to metadata
        checksum = hashlib.md5(audio_data).hexdigest()
        meta = metadata or {}
        meta["checksum"] = checksum
        meta["original_filename"] = filename
        meta["uploaded_at"] = datetime.utcnow().isoformat()

        url = await self._backend.upload(
            key=key,
            data=audio_data,
            content_type=content_type,
            metadata=meta,
        )

        return key, url

    async def get_sample(self, key: str) -> bytes:
        """Get an audio sample by key."""
        return await self._backend.download(key)

    async def get_sample_url(
        self,
        key: str,
        expires_in: int = 3600,
    ) -> str:
        """Get a URL for accessing a sample."""
        return await self._backend.get_url(key, expires_in)

    async def delete_sample(self, key: str) -> bool:
        """Delete a sample."""
        return await self._backend.delete(key)

    async def list_samples(self, tenant_id: str) -> List[str]:
        """List all samples for a tenant."""
        return await self._backend.list_keys(f"{tenant_id}/samples/")

    async def get_storage_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get storage usage statistics for a tenant."""
        samples = await self.list_samples(tenant_id)

        total_size = 0
        for key in samples:
            try:
                data = await self._backend.download(key)
                total_size += len(data)
            except Exception:
                pass

        return {
            "tenant_id": tenant_id,
            "sample_count": len(samples),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
        }

    async def cleanup_expired(
        self,
        tenant_id: str,
        max_age_days: int = 365,
    ) -> int:
        """
        Cleanup expired samples.

        Returns number of deleted samples.
        """
        # Implementation would check metadata for upload date
        # and delete samples older than max_age_days
        return 0

"""
Recording Storage Provider Module

This module provides storage backend implementations for managing call recordings
across different cloud and local storage systems.
"""

import asyncio
import hashlib
import io
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    BinaryIO,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from .base import (
    Recording,
    StorageConfig,
    StorageBackend,
    StorageError,
    RecordingFormat,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Storage Provider Interface
# =============================================================================


class StorageProvider(ABC):
    """Abstract base class for storage providers."""

    def __init__(self, config: StorageConfig):
        """
        Initialize storage provider.

        Args:
            config: Storage configuration
        """
        self.config = config

    @abstractmethod
    async def upload(
        self,
        data: Union[bytes, BinaryIO, AsyncIterator[bytes]],
        path: str,
        content_type: str = "audio/wav",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload data to storage.

        Args:
            data: Data to upload (bytes, file-like, or async iterator)
            path: Storage path
            content_type: MIME type
            metadata: Optional metadata

        Returns:
            Storage URL/path of uploaded file
        """
        pass

    @abstractmethod
    async def download(
        self,
        path: str,
    ) -> bytes:
        """
        Download data from storage.

        Args:
            path: Storage path

        Returns:
            Downloaded data as bytes
        """
        pass

    @abstractmethod
    async def download_stream(
        self,
        path: str,
        chunk_size: int = 8192,
    ) -> AsyncIterator[bytes]:
        """
        Download data as a stream.

        Args:
            path: Storage path
            chunk_size: Size of chunks to yield

        Yields:
            Chunks of data
        """
        pass

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """
        Delete file from storage.

        Args:
            path: Storage path

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """
        Check if file exists.

        Args:
            path: Storage path

        Returns:
            True if exists
        """
        pass

    @abstractmethod
    async def get_metadata(self, path: str) -> Dict[str, Any]:
        """
        Get file metadata.

        Args:
            path: Storage path

        Returns:
            File metadata
        """
        pass

    @abstractmethod
    async def generate_presigned_url(
        self,
        path: str,
        expiration_seconds: int = 3600,
        method: str = "GET",
    ) -> str:
        """
        Generate presigned URL for direct access.

        Args:
            path: Storage path
            expiration_seconds: URL expiration time
            method: HTTP method (GET, PUT)

        Returns:
            Presigned URL
        """
        pass

    @abstractmethod
    async def copy(
        self,
        source_path: str,
        dest_path: str,
    ) -> str:
        """
        Copy file within storage.

        Args:
            source_path: Source path
            dest_path: Destination path

        Returns:
            Destination URL/path
        """
        pass

    @abstractmethod
    async def list_files(
        self,
        prefix: str,
        max_results: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        List files with prefix.

        Args:
            prefix: Path prefix
            max_results: Maximum results

        Returns:
            List of file info dicts
        """
        pass

    def generate_path(
        self,
        recording: Recording,
        extension: Optional[str] = None,
    ) -> str:
        """
        Generate storage path for a recording.

        Args:
            recording: Recording object
            extension: File extension override

        Returns:
            Generated storage path
        """
        template = self.config.path_template
        now = recording.created_at

        ext = extension or recording.audio.format.value
        path = template.format(
            org_id=recording.organization_id,
            year=now.strftime("%Y"),
            month=now.strftime("%m"),
            day=now.strftime("%d"),
            call_id=recording.call_id,
            recording_id=recording.id,
        )

        full_path = f"{self.config.base_path}/{path}.{ext}"
        return full_path.replace("//", "/")

    @staticmethod
    def calculate_hash(data: bytes) -> str:
        """Calculate SHA-256 hash of data."""
        return hashlib.sha256(data).hexdigest()


# =============================================================================
# Local Storage Provider
# =============================================================================


class LocalStorageProvider(StorageProvider):
    """Local filesystem storage provider."""

    def __init__(self, config: StorageConfig):
        """Initialize local storage provider."""
        super().__init__(config)
        self.base_dir = Path(config.bucket_name or "/var/recordings")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_full_path(self, path: str) -> Path:
        """Get full filesystem path."""
        return self.base_dir / path

    async def upload(
        self,
        data: Union[bytes, BinaryIO, AsyncIterator[bytes]],
        path: str,
        content_type: str = "audio/wav",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload to local filesystem."""
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle different data types
        if isinstance(data, bytes):
            content = data
        elif hasattr(data, "read"):
            content = data.read()
        else:
            # Async iterator
            chunks = []
            async for chunk in data:
                chunks.append(chunk)
            content = b"".join(chunks)

        # Write file
        full_path.write_bytes(content)

        # Write metadata
        if metadata:
            meta_path = full_path.with_suffix(full_path.suffix + ".meta")
            import json
            meta_path.write_text(json.dumps(metadata))

        logger.info(f"Uploaded to local: {path}")
        return str(full_path)

    async def download(self, path: str) -> bytes:
        """Download from local filesystem."""
        full_path = self._get_full_path(path)
        if not full_path.exists():
            raise StorageError(f"File not found: {path}")
        return full_path.read_bytes()

    async def download_stream(
        self,
        path: str,
        chunk_size: int = 8192,
    ) -> AsyncIterator[bytes]:
        """Download as stream from local filesystem."""
        full_path = self._get_full_path(path)
        if not full_path.exists():
            raise StorageError(f"File not found: {path}")

        with open(full_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    async def delete(self, path: str) -> bool:
        """Delete from local filesystem."""
        full_path = self._get_full_path(path)
        if full_path.exists():
            full_path.unlink()
            # Also delete metadata
            meta_path = full_path.with_suffix(full_path.suffix + ".meta")
            if meta_path.exists():
                meta_path.unlink()
            logger.info(f"Deleted from local: {path}")
            return True
        return False

    async def exists(self, path: str) -> bool:
        """Check if exists in local filesystem."""
        return self._get_full_path(path).exists()

    async def get_metadata(self, path: str) -> Dict[str, Any]:
        """Get metadata from local filesystem."""
        full_path = self._get_full_path(path)
        if not full_path.exists():
            raise StorageError(f"File not found: {path}")

        stat = full_path.stat()
        metadata = {
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        }

        # Load custom metadata
        meta_path = full_path.with_suffix(full_path.suffix + ".meta")
        if meta_path.exists():
            import json
            custom_meta = json.loads(meta_path.read_text())
            metadata["custom"] = custom_meta

        return metadata

    async def generate_presigned_url(
        self,
        path: str,
        expiration_seconds: int = 3600,
        method: str = "GET",
    ) -> str:
        """Generate file URL (local paths don't have presigned URLs)."""
        return f"file://{self._get_full_path(path)}"

    async def copy(
        self,
        source_path: str,
        dest_path: str,
    ) -> str:
        """Copy file in local filesystem."""
        import shutil
        src = self._get_full_path(source_path)
        dst = self._get_full_path(dest_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst)

    async def list_files(
        self,
        prefix: str,
        max_results: int = 1000,
    ) -> List[Dict[str, Any]]:
        """List files with prefix in local filesystem."""
        base = self._get_full_path(prefix)
        files = []

        if base.is_dir():
            for path in base.rglob("*"):
                if path.is_file() and not path.suffix == ".meta":
                    stat = path.stat()
                    files.append({
                        "path": str(path.relative_to(self.base_dir)),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
                    if len(files) >= max_results:
                        break

        return files


# =============================================================================
# S3 Storage Provider
# =============================================================================


class S3StorageProvider(StorageProvider):
    """AWS S3 storage provider."""

    def __init__(self, config: StorageConfig):
        """Initialize S3 storage provider."""
        super().__init__(config)
        self._client = None
        self._session = None

    async def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            try:
                import aiobotocore.session
                self._session = aiobotocore.session.get_session()
                self._client = await self._session.create_client(
                    "s3",
                    region_name=self.config.region,
                    endpoint_url=self.config.endpoint_url,
                    aws_access_key_id=self.config.access_key,
                    aws_secret_access_key=self.config.secret_key,
                ).__aenter__()
            except ImportError:
                # Fallback to boto3 with sync-to-async wrapper
                import boto3
                self._client = boto3.client(
                    "s3",
                    region_name=self.config.region,
                    endpoint_url=self.config.endpoint_url,
                    aws_access_key_id=self.config.access_key,
                    aws_secret_access_key=self.config.secret_key,
                )
        return self._client

    async def upload(
        self,
        data: Union[bytes, BinaryIO, AsyncIterator[bytes]],
        path: str,
        content_type: str = "audio/wav",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload to S3."""
        client = await self._get_client()

        # Convert data to bytes if needed
        if isinstance(data, bytes):
            body = data
        elif hasattr(data, "read"):
            body = data.read()
        else:
            chunks = []
            async for chunk in data:
                chunks.append(chunk)
            body = b"".join(chunks)

        # Prepare upload params
        params = {
            "Bucket": self.config.bucket_name,
            "Key": path,
            "Body": body,
            "ContentType": content_type,
            "StorageClass": self.config.storage_class,
        }

        if metadata:
            params["Metadata"] = metadata

        if self.config.encryption_enabled:
            params["ServerSideEncryption"] = "AES256"
            if self.config.encryption_key_id:
                params["ServerSideEncryption"] = "aws:kms"
                params["SSEKMSKeyId"] = self.config.encryption_key_id

        # Check for multipart upload
        if len(body) > self.config.multipart_threshold_mb * 1024 * 1024:
            return await self._multipart_upload(body, path, params)

        # Regular upload
        if hasattr(client, "put_object"):
            await client.put_object(**params)
        else:
            # Sync client
            client.put_object(**params)

        logger.info(f"Uploaded to S3: {path}")
        return f"s3://{self.config.bucket_name}/{path}"

    async def _multipart_upload(
        self,
        data: bytes,
        path: str,
        params: Dict[str, Any],
    ) -> str:
        """Perform multipart upload for large files."""
        client = await self._get_client()
        chunk_size = self.config.multipart_chunk_size_mb * 1024 * 1024

        # Initiate multipart upload
        response = await client.create_multipart_upload(
            Bucket=self.config.bucket_name,
            Key=path,
            ContentType=params.get("ContentType", "audio/wav"),
            StorageClass=params.get("StorageClass", "STANDARD"),
        )
        upload_id = response["UploadId"]

        try:
            parts = []
            part_number = 1

            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                response = await client.upload_part(
                    Bucket=self.config.bucket_name,
                    Key=path,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=chunk,
                )
                parts.append({
                    "PartNumber": part_number,
                    "ETag": response["ETag"],
                })
                part_number += 1

            # Complete multipart upload
            await client.complete_multipart_upload(
                Bucket=self.config.bucket_name,
                Key=path,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )

            logger.info(f"Completed multipart upload to S3: {path}")
            return f"s3://{self.config.bucket_name}/{path}"

        except Exception as e:
            # Abort on failure
            await client.abort_multipart_upload(
                Bucket=self.config.bucket_name,
                Key=path,
                UploadId=upload_id,
            )
            raise StorageError(f"Multipart upload failed: {e}")

    async def download(self, path: str) -> bytes:
        """Download from S3."""
        client = await self._get_client()

        try:
            if hasattr(client, "get_object"):
                response = await client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=path,
                )
                async with response["Body"] as stream:
                    return await stream.read()
            else:
                response = client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=path,
                )
                return response["Body"].read()
        except Exception as e:
            raise StorageError(f"Failed to download from S3: {e}")

    async def download_stream(
        self,
        path: str,
        chunk_size: int = 8192,
    ) -> AsyncIterator[bytes]:
        """Download as stream from S3."""
        client = await self._get_client()

        try:
            response = await client.get_object(
                Bucket=self.config.bucket_name,
                Key=path,
            )
            async with response["Body"] as stream:
                while True:
                    chunk = await stream.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            raise StorageError(f"Failed to stream from S3: {e}")

    async def delete(self, path: str) -> bool:
        """Delete from S3."""
        client = await self._get_client()

        try:
            if hasattr(client, "delete_object"):
                await client.delete_object(
                    Bucket=self.config.bucket_name,
                    Key=path,
                )
            else:
                client.delete_object(
                    Bucket=self.config.bucket_name,
                    Key=path,
                )
            logger.info(f"Deleted from S3: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from S3: {e}")
            return False

    async def exists(self, path: str) -> bool:
        """Check if exists in S3."""
        client = await self._get_client()

        try:
            if hasattr(client, "head_object"):
                await client.head_object(
                    Bucket=self.config.bucket_name,
                    Key=path,
                )
            else:
                client.head_object(
                    Bucket=self.config.bucket_name,
                    Key=path,
                )
            return True
        except Exception:
            return False

    async def get_metadata(self, path: str) -> Dict[str, Any]:
        """Get metadata from S3."""
        client = await self._get_client()

        try:
            if hasattr(client, "head_object"):
                response = await client.head_object(
                    Bucket=self.config.bucket_name,
                    Key=path,
                )
            else:
                response = client.head_object(
                    Bucket=self.config.bucket_name,
                    Key=path,
                )

            return {
                "size": response.get("ContentLength", 0),
                "content_type": response.get("ContentType", ""),
                "modified": response.get("LastModified", datetime.utcnow()).isoformat(),
                "etag": response.get("ETag", "").strip('"'),
                "custom": response.get("Metadata", {}),
            }
        except Exception as e:
            raise StorageError(f"Failed to get metadata from S3: {e}")

    async def generate_presigned_url(
        self,
        path: str,
        expiration_seconds: int = 3600,
        method: str = "GET",
    ) -> str:
        """Generate presigned URL for S3."""
        client = await self._get_client()

        client_method = "get_object" if method == "GET" else "put_object"

        try:
            if hasattr(client, "generate_presigned_url"):
                url = await client.generate_presigned_url(
                    client_method,
                    Params={
                        "Bucket": self.config.bucket_name,
                        "Key": path,
                    },
                    ExpiresIn=expiration_seconds,
                )
            else:
                url = client.generate_presigned_url(
                    client_method,
                    Params={
                        "Bucket": self.config.bucket_name,
                        "Key": path,
                    },
                    ExpiresIn=expiration_seconds,
                )
            return url
        except Exception as e:
            raise StorageError(f"Failed to generate presigned URL: {e}")

    async def copy(
        self,
        source_path: str,
        dest_path: str,
    ) -> str:
        """Copy file within S3."""
        client = await self._get_client()

        try:
            copy_source = {
                "Bucket": self.config.bucket_name,
                "Key": source_path,
            }
            if hasattr(client, "copy_object"):
                await client.copy_object(
                    Bucket=self.config.bucket_name,
                    Key=dest_path,
                    CopySource=copy_source,
                )
            else:
                client.copy_object(
                    Bucket=self.config.bucket_name,
                    Key=dest_path,
                    CopySource=copy_source,
                )
            return f"s3://{self.config.bucket_name}/{dest_path}"
        except Exception as e:
            raise StorageError(f"Failed to copy in S3: {e}")

    async def list_files(
        self,
        prefix: str,
        max_results: int = 1000,
    ) -> List[Dict[str, Any]]:
        """List files with prefix in S3."""
        client = await self._get_client()
        files = []

        try:
            if hasattr(client, "list_objects_v2"):
                paginator = client.get_paginator("list_objects_v2")
                async for page in paginator.paginate(
                    Bucket=self.config.bucket_name,
                    Prefix=prefix,
                    MaxKeys=max_results,
                ):
                    for obj in page.get("Contents", []):
                        files.append({
                            "path": obj["Key"],
                            "size": obj["Size"],
                            "modified": obj["LastModified"].isoformat(),
                            "etag": obj["ETag"].strip('"'),
                        })
                        if len(files) >= max_results:
                            return files
            else:
                response = client.list_objects_v2(
                    Bucket=self.config.bucket_name,
                    Prefix=prefix,
                    MaxKeys=max_results,
                )
                for obj in response.get("Contents", []):
                    files.append({
                        "path": obj["Key"],
                        "size": obj["Size"],
                        "modified": obj["LastModified"].isoformat(),
                        "etag": obj["ETag"].strip('"'),
                    })
        except Exception as e:
            raise StorageError(f"Failed to list files in S3: {e}")

        return files


# =============================================================================
# GCS Storage Provider
# =============================================================================


class GCSStorageProvider(StorageProvider):
    """Google Cloud Storage provider."""

    def __init__(self, config: StorageConfig):
        """Initialize GCS storage provider."""
        super().__init__(config)
        self._client = None

    async def _get_client(self):
        """Get or create GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client()
            except ImportError:
                raise StorageError("google-cloud-storage package not installed")
        return self._client

    async def _get_bucket(self):
        """Get bucket object."""
        client = await self._get_client()
        return client.bucket(self.config.bucket_name)

    async def upload(
        self,
        data: Union[bytes, BinaryIO, AsyncIterator[bytes]],
        path: str,
        content_type: str = "audio/wav",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload to GCS."""
        bucket = await self._get_bucket()
        blob = bucket.blob(path)

        # Convert data to bytes if needed
        if isinstance(data, bytes):
            content = data
        elif hasattr(data, "read"):
            content = data.read()
        else:
            chunks = []
            async for chunk in data:
                chunks.append(chunk)
            content = b"".join(chunks)

        blob.upload_from_string(
            content,
            content_type=content_type,
        )

        if metadata:
            blob.metadata = metadata
            blob.patch()

        logger.info(f"Uploaded to GCS: {path}")
        return f"gs://{self.config.bucket_name}/{path}"

    async def download(self, path: str) -> bytes:
        """Download from GCS."""
        bucket = await self._get_bucket()
        blob = bucket.blob(path)

        if not blob.exists():
            raise StorageError(f"File not found: {path}")

        return blob.download_as_bytes()

    async def download_stream(
        self,
        path: str,
        chunk_size: int = 8192,
    ) -> AsyncIterator[bytes]:
        """Download as stream from GCS."""
        bucket = await self._get_bucket()
        blob = bucket.blob(path)

        if not blob.exists():
            raise StorageError(f"File not found: {path}")

        # Download to bytes and yield in chunks
        data = blob.download_as_bytes()
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    async def delete(self, path: str) -> bool:
        """Delete from GCS."""
        bucket = await self._get_bucket()
        blob = bucket.blob(path)

        try:
            blob.delete()
            logger.info(f"Deleted from GCS: {path}")
            return True
        except Exception:
            return False

    async def exists(self, path: str) -> bool:
        """Check if exists in GCS."""
        bucket = await self._get_bucket()
        blob = bucket.blob(path)
        return blob.exists()

    async def get_metadata(self, path: str) -> Dict[str, Any]:
        """Get metadata from GCS."""
        bucket = await self._get_bucket()
        blob = bucket.blob(path)

        if not blob.exists():
            raise StorageError(f"File not found: {path}")

        blob.reload()
        return {
            "size": blob.size,
            "content_type": blob.content_type,
            "modified": blob.updated.isoformat() if blob.updated else None,
            "crc32c": blob.crc32c,
            "md5": blob.md5_hash,
            "custom": blob.metadata or {},
        }

    async def generate_presigned_url(
        self,
        path: str,
        expiration_seconds: int = 3600,
        method: str = "GET",
    ) -> str:
        """Generate signed URL for GCS."""
        bucket = await self._get_bucket()
        blob = bucket.blob(path)

        return blob.generate_signed_url(
            expiration=timedelta(seconds=expiration_seconds),
            method=method,
        )

    async def copy(
        self,
        source_path: str,
        dest_path: str,
    ) -> str:
        """Copy file within GCS."""
        bucket = await self._get_bucket()
        source_blob = bucket.blob(source_path)
        dest_blob = bucket.blob(dest_path)

        bucket.copy_blob(source_blob, bucket, dest_path)
        return f"gs://{self.config.bucket_name}/{dest_path}"

    async def list_files(
        self,
        prefix: str,
        max_results: int = 1000,
    ) -> List[Dict[str, Any]]:
        """List files with prefix in GCS."""
        bucket = await self._get_bucket()
        blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)

        files = []
        for blob in blobs:
            files.append({
                "path": blob.name,
                "size": blob.size,
                "modified": blob.updated.isoformat() if blob.updated else None,
                "crc32c": blob.crc32c,
            })

        return files


# =============================================================================
# Storage Provider Factory
# =============================================================================


class StorageProviderFactory:
    """Factory for creating storage providers."""

    _providers: Dict[StorageBackend, type] = {
        StorageBackend.LOCAL: LocalStorageProvider,
        StorageBackend.S3: S3StorageProvider,
        StorageBackend.GCS: GCSStorageProvider,
        StorageBackend.MINIO: S3StorageProvider,  # MinIO is S3 compatible
    }

    @classmethod
    def create(cls, config: StorageConfig) -> StorageProvider:
        """
        Create storage provider from configuration.

        Args:
            config: Storage configuration

        Returns:
            Storage provider instance
        """
        provider_class = cls._providers.get(config.backend)
        if not provider_class:
            raise StorageError(f"Unknown storage backend: {config.backend}")

        return provider_class(config)

    @classmethod
    def register(
        cls,
        backend: StorageBackend,
        provider_class: type,
    ) -> None:
        """
        Register a custom storage provider.

        Args:
            backend: Storage backend type
            provider_class: Provider class
        """
        cls._providers[backend] = provider_class


# =============================================================================
# Storage Manager
# =============================================================================


class StorageManager:
    """
    Manages recording storage across multiple backends.
    """

    def __init__(
        self,
        default_config: Optional[StorageConfig] = None,
    ):
        """
        Initialize storage manager.

        Args:
            default_config: Default storage configuration
        """
        self._providers: Dict[str, StorageProvider] = {}
        self._default_provider: Optional[StorageProvider] = None

        if default_config:
            self._default_provider = StorageProviderFactory.create(default_config)

    def add_provider(
        self,
        name: str,
        config: StorageConfig,
        is_default: bool = False,
    ) -> StorageProvider:
        """
        Add a storage provider.

        Args:
            name: Provider name
            config: Storage configuration
            is_default: Set as default provider

        Returns:
            Created provider
        """
        provider = StorageProviderFactory.create(config)
        self._providers[name] = provider

        if is_default:
            self._default_provider = provider

        return provider

    def get_provider(self, name: Optional[str] = None) -> StorageProvider:
        """
        Get a storage provider.

        Args:
            name: Provider name (uses default if None)

        Returns:
            Storage provider
        """
        if name:
            if name not in self._providers:
                raise StorageError(f"Provider not found: {name}")
            return self._providers[name]

        if not self._default_provider:
            raise StorageError("No default storage provider configured")

        return self._default_provider

    async def upload_recording(
        self,
        recording: Recording,
        data: Union[bytes, BinaryIO, AsyncIterator[bytes]],
        provider_name: Optional[str] = None,
    ) -> Recording:
        """
        Upload a recording to storage.

        Args:
            recording: Recording object
            data: Recording data
            provider_name: Storage provider to use

        Returns:
            Updated recording with storage info
        """
        provider = self.get_provider(provider_name)

        # Generate path
        path = provider.generate_path(recording)

        # Determine content type
        content_type = f"audio/{recording.audio.format.value}"

        # Upload
        storage_url = await provider.upload(
            data=data,
            path=path,
            content_type=content_type,
            metadata={
                "recording_id": recording.id,
                "call_id": recording.call_id,
                "organization_id": recording.organization_id,
            },
        )

        # Calculate hash if bytes
        if isinstance(data, bytes):
            recording.content_hash = StorageProvider.calculate_hash(data)
            recording.audio.file_size_bytes = len(data)

        # Update recording
        recording.storage_path = path
        recording.storage_bucket = provider.config.bucket_name
        recording.storage_backend = provider.config.backend
        recording.updated_at = datetime.utcnow()

        return recording

    async def download_recording(
        self,
        recording: Recording,
        provider_name: Optional[str] = None,
    ) -> bytes:
        """
        Download a recording from storage.

        Args:
            recording: Recording object
            provider_name: Storage provider to use

        Returns:
            Recording data as bytes
        """
        provider = self.get_provider(provider_name)
        return await provider.download(recording.storage_path)

    async def get_download_url(
        self,
        recording: Recording,
        expiration_seconds: int = 3600,
        provider_name: Optional[str] = None,
    ) -> str:
        """
        Get presigned download URL for a recording.

        Args:
            recording: Recording object
            expiration_seconds: URL expiration time
            provider_name: Storage provider to use

        Returns:
            Presigned URL
        """
        provider = self.get_provider(provider_name)
        return await provider.generate_presigned_url(
            recording.storage_path,
            expiration_seconds=expiration_seconds,
            method="GET",
        )

    async def delete_recording(
        self,
        recording: Recording,
        provider_name: Optional[str] = None,
    ) -> bool:
        """
        Delete a recording from storage.

        Args:
            recording: Recording object
            provider_name: Storage provider to use

        Returns:
            True if deleted
        """
        provider = self.get_provider(provider_name)
        return await provider.delete(recording.storage_path)

    async def archive_recording(
        self,
        recording: Recording,
        archive_provider_name: str,
        source_provider_name: Optional[str] = None,
    ) -> Recording:
        """
        Archive a recording to a different storage tier.

        Args:
            recording: Recording object
            archive_provider_name: Archive storage provider
            source_provider_name: Source provider

        Returns:
            Updated recording
        """
        source = self.get_provider(source_provider_name)
        archive = self.get_provider(archive_provider_name)

        # Download from source
        data = await source.download(recording.storage_path)

        # Generate archive path
        archive_path = archive.generate_path(recording)

        # Upload to archive
        await archive.upload(
            data=data,
            path=archive_path,
            content_type=f"audio/{recording.audio.format.value}",
        )

        # Update recording
        recording.storage_path = archive_path
        recording.storage_bucket = archive.config.bucket_name
        recording.storage_backend = archive.config.backend
        recording.archived_at = datetime.utcnow()
        recording.updated_at = datetime.utcnow()

        # Delete from source
        await source.delete(recording.storage_path)

        return recording


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "StorageProvider",
    "LocalStorageProvider",
    "S3StorageProvider",
    "GCSStorageProvider",
    "StorageProviderFactory",
    "StorageManager",
]

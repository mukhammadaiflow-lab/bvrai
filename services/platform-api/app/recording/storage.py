"""Storage backends for recordings."""

from typing import Optional, Dict, Any, List, BinaryIO, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pathlib import Path
import asyncio
import aiofiles
import logging
import os
import hashlib
import mimetypes

logger = logging.getLogger(__name__)


@dataclass
class StorageObject:
    """Metadata for stored object."""
    key: str
    size: int
    content_type: str
    etag: Optional[str] = None
    last_modified: Optional[datetime] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "size": self.size,
            "content_type": self.content_type,
            "etag": self.etag,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "metadata": self.metadata,
            "url": self.url,
        }


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def upload(
        self,
        data: bytes,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """Upload data to storage."""
        pass

    @abstractmethod
    async def upload_file(
        self,
        file_path: str,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """Upload file to storage."""
        pass

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Download data from storage."""
        pass

    @abstractmethod
    async def download_file(self, key: str, file_path: str) -> None:
        """Download to file."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete object from storage."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if object exists."""
        pass

    @abstractmethod
    async def get_metadata(self, key: str) -> Optional[StorageObject]:
        """Get object metadata."""
        pass

    @abstractmethod
    async def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> List[StorageObject]:
        """List objects with prefix."""
        pass

    @abstractmethod
    async def get_signed_url(
        self,
        key: str,
        expires_in: int = 3600,
        method: str = "GET",
    ) -> str:
        """Get signed URL for object."""
        pass


class LocalStorage(StorageBackend):
    """
    Local filesystem storage backend.

    Usage:
        storage = LocalStorage(base_path="/data/recordings")
        await storage.upload(audio_data, "calls/123/recording.wav")
    """

    def __init__(
        self,
        base_path: str = "/tmp/storage",
        base_url: Optional[str] = None,
    ):
        self.base_path = Path(base_path)
        self.base_url = base_url or f"file://{base_path}"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_full_path(self, key: str) -> Path:
        """Get full path for key."""
        return self.base_path / key

    def _get_content_type(self, key: str) -> str:
        """Get content type for key."""
        mime_type, _ = mimetypes.guess_type(key)
        return mime_type or "application/octet-stream"

    async def upload(
        self,
        data: bytes,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """Upload data to local storage."""
        file_path = self._get_full_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)

        etag = hashlib.md5(data).hexdigest()

        return StorageObject(
            key=key,
            size=len(data),
            content_type=content_type or self._get_content_type(key),
            etag=etag,
            last_modified=datetime.utcnow(),
            metadata=metadata or {},
            url=f"{self.base_url}/{key}",
        )

    async def upload_file(
        self,
        file_path: str,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """Upload file to local storage."""
        async with aiofiles.open(file_path, 'rb') as f:
            data = await f.read()
        return await self.upload(data, key, content_type, metadata)

    async def download(self, key: str) -> bytes:
        """Download from local storage."""
        file_path = self._get_full_path(key)
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()

    async def download_file(self, key: str, file_path: str) -> None:
        """Download to file."""
        data = await self.download(key)
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)

    async def delete(self, key: str) -> bool:
        """Delete from local storage."""
        file_path = self._get_full_path(key)
        try:
            file_path.unlink()
            return True
        except FileNotFoundError:
            return False

    async def exists(self, key: str) -> bool:
        """Check if exists in local storage."""
        return self._get_full_path(key).exists()

    async def get_metadata(self, key: str) -> Optional[StorageObject]:
        """Get metadata from local storage."""
        file_path = self._get_full_path(key)
        if not file_path.exists():
            return None

        stat = file_path.stat()
        return StorageObject(
            key=key,
            size=stat.st_size,
            content_type=self._get_content_type(key),
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            url=f"{self.base_url}/{key}",
        )

    async def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> List[StorageObject]:
        """List objects in local storage."""
        base = self._get_full_path(prefix)
        objects = []

        if not base.exists():
            return objects

        for path in base.rglob("*"):
            if path.is_file():
                key = str(path.relative_to(self.base_path))
                stat = path.stat()
                objects.append(StorageObject(
                    key=key,
                    size=stat.st_size,
                    content_type=self._get_content_type(key),
                    last_modified=datetime.fromtimestamp(stat.st_mtime),
                    url=f"{self.base_url}/{key}",
                ))

                if len(objects) >= max_keys:
                    break

        return objects

    async def get_signed_url(
        self,
        key: str,
        expires_in: int = 3600,
        method: str = "GET",
    ) -> str:
        """Get URL for local file."""
        return f"{self.base_url}/{key}"


class S3Storage(StorageBackend):
    """
    AWS S3 storage backend.

    Usage:
        storage = S3Storage(
            bucket="my-recordings",
            region="us-east-1",
        )
        await storage.upload(audio_data, "calls/123/recording.wav")
    """

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        self.bucket = bucket
        self.region = region
        self.access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID")
        self.secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.endpoint_url = endpoint_url

        self._client = None

    async def _get_client(self):
        """Get S3 client."""
        if self._client is None:
            try:
                import aioboto3
                session = aioboto3.Session(
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name=self.region,
                )
                self._client = await session.client(
                    's3',
                    endpoint_url=self.endpoint_url,
                ).__aenter__()
            except ImportError:
                raise ImportError("aioboto3 is required for S3 storage")
        return self._client

    async def upload(
        self,
        data: bytes,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """Upload to S3."""
        client = await self._get_client()

        content_type = content_type or "application/octet-stream"
        extra_args = {
            "ContentType": content_type,
        }
        if metadata:
            extra_args["Metadata"] = metadata

        await client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            **extra_args,
        )

        etag = hashlib.md5(data).hexdigest()

        return StorageObject(
            key=key,
            size=len(data),
            content_type=content_type,
            etag=etag,
            last_modified=datetime.utcnow(),
            metadata=metadata or {},
            url=f"s3://{self.bucket}/{key}",
        )

    async def upload_file(
        self,
        file_path: str,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """Upload file to S3."""
        async with aiofiles.open(file_path, 'rb') as f:
            data = await f.read()
        return await self.upload(data, key, content_type, metadata)

    async def download(self, key: str) -> bytes:
        """Download from S3."""
        client = await self._get_client()

        response = await client.get_object(
            Bucket=self.bucket,
            Key=key,
        )

        async with response['Body'] as stream:
            return await stream.read()

    async def download_file(self, key: str, file_path: str) -> None:
        """Download to file from S3."""
        data = await self.download(key)
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)

    async def delete(self, key: str) -> bool:
        """Delete from S3."""
        client = await self._get_client()
        try:
            await client.delete_object(
                Bucket=self.bucket,
                Key=key,
            )
            return True
        except Exception as e:
            logger.error(f"S3 delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if exists in S3."""
        client = await self._get_client()
        try:
            await client.head_object(
                Bucket=self.bucket,
                Key=key,
            )
            return True
        except Exception:
            return False

    async def get_metadata(self, key: str) -> Optional[StorageObject]:
        """Get metadata from S3."""
        client = await self._get_client()
        try:
            response = await client.head_object(
                Bucket=self.bucket,
                Key=key,
            )
            return StorageObject(
                key=key,
                size=response.get('ContentLength', 0),
                content_type=response.get('ContentType', 'application/octet-stream'),
                etag=response.get('ETag', '').strip('"'),
                last_modified=response.get('LastModified'),
                metadata=response.get('Metadata', {}),
                url=f"s3://{self.bucket}/{key}",
            )
        except Exception:
            return None

    async def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> List[StorageObject]:
        """List objects in S3."""
        client = await self._get_client()
        objects = []

        paginator = client.get_paginator('list_objects_v2')
        async for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix,
            MaxKeys=max_keys,
        ):
            for obj in page.get('Contents', []):
                objects.append(StorageObject(
                    key=obj['Key'],
                    size=obj['Size'],
                    content_type="application/octet-stream",
                    etag=obj.get('ETag', '').strip('"'),
                    last_modified=obj.get('LastModified'),
                    url=f"s3://{self.bucket}/{obj['Key']}",
                ))

        return objects[:max_keys]

    async def get_signed_url(
        self,
        key: str,
        expires_in: int = 3600,
        method: str = "GET",
    ) -> str:
        """Get signed URL for S3 object."""
        client = await self._get_client()

        client_method = 'get_object' if method == 'GET' else 'put_object'

        url = await client.generate_presigned_url(
            ClientMethod=client_method,
            Params={
                'Bucket': self.bucket,
                'Key': key,
            },
            ExpiresIn=expires_in,
        )

        return url


class GCSStorage(StorageBackend):
    """
    Google Cloud Storage backend.

    Usage:
        storage = GCSStorage(bucket="my-recordings")
        await storage.upload(audio_data, "calls/123/recording.wav")
    """

    def __init__(
        self,
        bucket: str,
        credentials_path: Optional[str] = None,
    ):
        self.bucket = bucket
        self.credentials_path = credentials_path

        self._client = None
        self._bucket_obj = None

    async def _get_bucket(self):
        """Get GCS bucket."""
        if self._bucket_obj is None:
            try:
                from google.cloud import storage
                if self.credentials_path:
                    self._client = storage.Client.from_service_account_json(
                        self.credentials_path
                    )
                else:
                    self._client = storage.Client()
                self._bucket_obj = self._client.bucket(self.bucket)
            except ImportError:
                raise ImportError("google-cloud-storage is required for GCS storage")
        return self._bucket_obj

    async def upload(
        self,
        data: bytes,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """Upload to GCS."""
        bucket = await self._get_bucket()
        blob = bucket.blob(key)

        if content_type:
            blob.content_type = content_type
        if metadata:
            blob.metadata = metadata

        # Run in executor since google-cloud-storage is not async
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: blob.upload_from_string(data, content_type=content_type),
        )

        return StorageObject(
            key=key,
            size=len(data),
            content_type=content_type or "application/octet-stream",
            etag=blob.etag,
            last_modified=datetime.utcnow(),
            metadata=metadata or {},
            url=f"gs://{self.bucket}/{key}",
        )

    async def upload_file(
        self,
        file_path: str,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """Upload file to GCS."""
        async with aiofiles.open(file_path, 'rb') as f:
            data = await f.read()
        return await self.upload(data, key, content_type, metadata)

    async def download(self, key: str) -> bytes:
        """Download from GCS."""
        bucket = await self._get_bucket()
        blob = bucket.blob(key)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            blob.download_as_bytes,
        )

    async def download_file(self, key: str, file_path: str) -> None:
        """Download to file from GCS."""
        data = await self.download(key)
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)

    async def delete(self, key: str) -> bool:
        """Delete from GCS."""
        bucket = await self._get_bucket()
        blob = bucket.blob(key)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, blob.delete)
            return True
        except Exception as e:
            logger.error(f"GCS delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if exists in GCS."""
        bucket = await self._get_bucket()
        blob = bucket.blob(key)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, blob.exists)

    async def get_metadata(self, key: str) -> Optional[StorageObject]:
        """Get metadata from GCS."""
        bucket = await self._get_bucket()
        blob = bucket.blob(key)

        loop = asyncio.get_event_loop()
        exists = await loop.run_in_executor(None, blob.exists)

        if not exists:
            return None

        await loop.run_in_executor(None, blob.reload)

        return StorageObject(
            key=key,
            size=blob.size or 0,
            content_type=blob.content_type or "application/octet-stream",
            etag=blob.etag,
            last_modified=blob.updated,
            metadata=blob.metadata or {},
            url=f"gs://{self.bucket}/{key}",
        )

    async def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> List[StorageObject]:
        """List objects in GCS."""
        bucket = await self._get_bucket()

        loop = asyncio.get_event_loop()
        blobs = await loop.run_in_executor(
            None,
            lambda: list(bucket.list_blobs(prefix=prefix, max_results=max_keys)),
        )

        return [
            StorageObject(
                key=blob.name,
                size=blob.size or 0,
                content_type=blob.content_type or "application/octet-stream",
                etag=blob.etag,
                last_modified=blob.updated,
                metadata=blob.metadata or {},
                url=f"gs://{self.bucket}/{blob.name}",
            )
            for blob in blobs
        ]

    async def get_signed_url(
        self,
        key: str,
        expires_in: int = 3600,
        method: str = "GET",
    ) -> str:
        """Get signed URL for GCS object."""
        bucket = await self._get_bucket()
        blob = bucket.blob(key)

        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(
            None,
            lambda: blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expires_in),
                method=method,
            ),
        )

        return url


class StorageManager:
    """
    Storage manager with support for multiple backends.

    Usage:
        manager = StorageManager()
        manager.add_backend("recordings", S3Storage(bucket="recordings"))
        manager.add_backend("temp", LocalStorage(base_path="/tmp"))

        await manager.upload("recordings", data, "call-123/audio.wav")
    """

    def __init__(self, default_backend: Optional[str] = None):
        self._backends: Dict[str, StorageBackend] = {}
        self._default_backend = default_backend

    def add_backend(self, name: str, backend: StorageBackend) -> None:
        """Add a storage backend."""
        self._backends[name] = backend
        if self._default_backend is None:
            self._default_backend = name

    def get_backend(self, name: Optional[str] = None) -> StorageBackend:
        """Get a storage backend."""
        name = name or self._default_backend
        if name is None or name not in self._backends:
            raise ValueError(f"Storage backend not found: {name}")
        return self._backends[name]

    async def upload(
        self,
        data: bytes,
        key: str,
        backend: Optional[str] = None,
        **kwargs,
    ) -> StorageObject:
        """Upload to storage."""
        storage = self.get_backend(backend)
        return await storage.upload(data, key, **kwargs)

    async def upload_file(
        self,
        file_path: str,
        key: str,
        backend: Optional[str] = None,
        **kwargs,
    ) -> StorageObject:
        """Upload file to storage."""
        storage = self.get_backend(backend)
        return await storage.upload_file(file_path, key, **kwargs)

    async def download(
        self,
        key: str,
        backend: Optional[str] = None,
    ) -> bytes:
        """Download from storage."""
        storage = self.get_backend(backend)
        return await storage.download(key)

    async def delete(
        self,
        key: str,
        backend: Optional[str] = None,
    ) -> bool:
        """Delete from storage."""
        storage = self.get_backend(backend)
        return await storage.delete(key)

    async def get_signed_url(
        self,
        key: str,
        backend: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Get signed URL."""
        storage = self.get_backend(backend)
        return await storage.get_signed_url(key, **kwargs)


# Global storage manager
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """Get or create the global storage manager."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
        # Add default local storage
        _storage_manager.add_backend(
            "local",
            LocalStorage(base_path="/tmp/bvrai-storage"),
        )
    return _storage_manager


def setup_storage(
    s3_bucket: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    local_path: str = "/tmp/bvrai-storage",
) -> StorageManager:
    """Setup storage backends."""
    manager = get_storage_manager()

    # Always add local storage
    manager.add_backend("local", LocalStorage(base_path=local_path))

    # Add S3 if configured
    if s3_bucket:
        manager.add_backend("s3", S3Storage(bucket=s3_bucket))

    # Add GCS if configured
    if gcs_bucket:
        manager.add_backend("gcs", GCSStorage(bucket=gcs_bucket))

    return manager

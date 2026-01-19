"""
AWS S3 Storage Integration

This module provides AWS S3 integration for the voice agent platform,
enabling file storage, retrieval, and management for call recordings,
transcripts, and other assets.

Features:
- File upload with multipart support
- Presigned URL generation for secure access
- Bucket management and lifecycle policies
- Object versioning support
- Server-side encryption
- Transfer acceleration
"""

import asyncio
import hashlib
import hmac
import logging
import mimetypes
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode
from io import BytesIO
import base64

import aiohttp

from ..base import (
    StorageProvider,
    StorageObject,
    AuthType,
    IntegrationStatus,
    IntegrationError,
    AuthenticationError,
)


logger = logging.getLogger(__name__)


class S3Provider(StorageProvider):
    """
    AWS S3 storage integration provider.

    Implements comprehensive S3 functionality including:
    - Object CRUD operations
    - Multipart uploads for large files
    - Presigned URLs for secure sharing
    - Server-side encryption (SSE-S3, SSE-KMS)
    - Object versioning
    - Lifecycle management

    AWS Signature Version 4 is used for authentication.
    """

    PROVIDER_NAME = "s3"
    AUTH_TYPE = AuthType.API_KEY  # AWS credentials

    # AWS configuration
    DEFAULT_REGION = "us-east-1"
    SERVICE = "s3"
    ALGORITHM = "AWS4-HMAC-SHA256"

    # Upload configuration
    MULTIPART_THRESHOLD = 100 * 1024 * 1024  # 100 MB
    MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_SINGLE_UPLOAD = 5 * 1024 * 1024 * 1024  # 5 GB

    # Presigned URL defaults
    DEFAULT_PRESIGNED_EXPIRY = 3600  # 1 hour
    MAX_PRESIGNED_EXPIRY = 604800  # 7 days

    def __init__(
        self,
        integration_id: str,
        organization_id: str,
        credentials: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize S3 provider.

        Args:
            integration_id: Unique integration identifier
            organization_id: Organization this integration belongs to
            credentials: AWS credentials including:
                - access_key_id: AWS access key ID
                - secret_access_key: AWS secret access key
                - session_token: Optional session token for temporary credentials
            settings: Optional settings including:
                - bucket: Default bucket name
                - region: AWS region (default: us-east-1)
                - endpoint_url: Custom endpoint for S3-compatible storage
                - path_style: Use path-style addressing (default: False)
                - encryption: Server-side encryption (AES256, aws:kms)
                - kms_key_id: KMS key ID for SSE-KMS
                - storage_class: Default storage class
                - transfer_acceleration: Enable transfer acceleration
        """
        super().__init__(integration_id, organization_id, credentials, settings or {})

        self._access_key_id = credentials.get("access_key_id")
        self._secret_access_key = credentials.get("secret_access_key")
        self._session_token = credentials.get("session_token")

        # Settings
        self._bucket = self.settings.get("bucket")
        self._region = self.settings.get("region", self.DEFAULT_REGION)
        self._endpoint_url = self.settings.get("endpoint_url")
        self._path_style = self.settings.get("path_style", False)
        self._encryption = self.settings.get("encryption")  # AES256 or aws:kms
        self._kms_key_id = self.settings.get("kms_key_id")
        self._storage_class = self.settings.get("storage_class", "STANDARD")
        self._transfer_acceleration = self.settings.get("transfer_acceleration", False)

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)  # 5 min timeout for uploads
            )
        return self._session

    def _get_endpoint(self, bucket: Optional[str] = None) -> str:
        """Get S3 endpoint URL."""
        if self._endpoint_url:
            return self._endpoint_url

        bucket = bucket or self._bucket

        if self._transfer_acceleration and bucket:
            return f"https://{bucket}.s3-accelerate.amazonaws.com"
        elif self._path_style or not bucket:
            return f"https://s3.{self._region}.amazonaws.com"
        else:
            return f"https://{bucket}.s3.{self._region}.amazonaws.com"

    def _get_host(self, bucket: Optional[str] = None) -> str:
        """Get S3 host for signing."""
        bucket = bucket or self._bucket

        if self._endpoint_url:
            from urllib.parse import urlparse
            return urlparse(self._endpoint_url).netloc
        elif self._transfer_acceleration and bucket:
            return f"{bucket}.s3-accelerate.amazonaws.com"
        elif self._path_style or not bucket:
            return f"s3.{self._region}.amazonaws.com"
        else:
            return f"{bucket}.s3.{self._region}.amazonaws.com"

    def _sign(self, key: bytes, msg: str) -> bytes:
        """HMAC-SHA256 sign a message."""
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    def _get_signature_key(self, date_stamp: str) -> bytes:
        """Get AWS Signature Version 4 signing key."""
        k_date = self._sign(
            f"AWS4{self._secret_access_key}".encode("utf-8"),
            date_stamp
        )
        k_region = self._sign(k_date, self._region)
        k_service = self._sign(k_region, self.SERVICE)
        k_signing = self._sign(k_service, "aws4_request")
        return k_signing

    def _create_canonical_request(
        self,
        method: str,
        path: str,
        query_params: Dict[str, str],
        headers: Dict[str, str],
        payload_hash: str,
    ) -> str:
        """Create canonical request for signing."""
        # Canonical URI
        canonical_uri = quote(path, safe="/")

        # Canonical query string
        sorted_params = sorted(query_params.items())
        canonical_querystring = "&".join(
            f"{quote(k, safe='')}={quote(v, safe='')}" for k, v in sorted_params
        )

        # Canonical headers
        sorted_headers = sorted(headers.items(), key=lambda x: x[0].lower())
        canonical_headers = "\n".join(
            f"{k.lower()}:{v.strip()}" for k, v in sorted_headers
        ) + "\n"

        # Signed headers
        signed_headers = ";".join(k.lower() for k, _ in sorted_headers)

        return "\n".join([
            method,
            canonical_uri,
            canonical_querystring,
            canonical_headers,
            signed_headers,
            payload_hash,
        ])

    def _create_string_to_sign(
        self,
        amz_date: str,
        date_stamp: str,
        canonical_request_hash: str,
    ) -> str:
        """Create string to sign."""
        credential_scope = f"{date_stamp}/{self._region}/{self.SERVICE}/aws4_request"
        return "\n".join([
            self.ALGORITHM,
            amz_date,
            credential_scope,
            canonical_request_hash,
        ])

    def _get_authorization_header(
        self,
        method: str,
        path: str,
        query_params: Dict[str, str],
        headers: Dict[str, str],
        payload_hash: str,
    ) -> Tuple[str, str, str]:
        """Generate AWS Signature Version 4 authorization header."""
        # Current time
        now = datetime.utcnow()
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")

        # Create canonical request
        canonical_request = self._create_canonical_request(
            method, path, query_params, headers, payload_hash
        )
        canonical_request_hash = hashlib.sha256(
            canonical_request.encode("utf-8")
        ).hexdigest()

        # Create string to sign
        string_to_sign = self._create_string_to_sign(
            amz_date, date_stamp, canonical_request_hash
        )

        # Calculate signature
        signing_key = self._get_signature_key(date_stamp)
        signature = hmac.new(
            signing_key,
            string_to_sign.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        # Build authorization header
        signed_headers = ";".join(sorted(k.lower() for k in headers.keys()))
        credential_scope = f"{date_stamp}/{self._region}/{self.SERVICE}/aws4_request"

        authorization = (
            f"{self.ALGORITHM} "
            f"Credential={self._access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )

        return authorization, amz_date, date_stamp

    async def _request(
        self,
        method: str,
        path: str,
        bucket: Optional[str] = None,
        query_params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[bytes] = None,
        stream_response: bool = False,
    ) -> aiohttp.ClientResponse:
        """
        Make an authenticated request to S3.

        Args:
            method: HTTP method
            path: Object path (key)
            bucket: Bucket name (uses default if not provided)
            query_params: Query parameters
            headers: Additional headers
            body: Request body
            stream_response: Return response for streaming

        Returns:
            Response object

        Raises:
            AuthenticationError: If authentication fails
            IntegrationError: For other errors
        """
        bucket = bucket or self._bucket
        query_params = query_params or {}
        headers = headers or {}

        # Calculate payload hash
        if body:
            payload_hash = hashlib.sha256(body).hexdigest()
        else:
            payload_hash = hashlib.sha256(b"").hexdigest()

        # Build URL path
        if self._path_style:
            url_path = f"/{bucket}{path}" if bucket else path
        else:
            url_path = path

        # Add required headers
        host = self._get_host(bucket)
        headers["Host"] = host
        headers["x-amz-content-sha256"] = payload_hash

        if self._session_token:
            headers["x-amz-security-token"] = self._session_token

        # Get authorization
        auth_header, amz_date, _ = self._get_authorization_header(
            method, url_path, query_params, headers, payload_hash
        )
        headers["x-amz-date"] = amz_date
        headers["Authorization"] = auth_header

        # Build URL
        endpoint = self._get_endpoint(bucket)
        if self._path_style:
            url = f"{endpoint}/{bucket}{path}" if bucket else f"{endpoint}{path}"
        else:
            url = f"{endpoint}{path}"

        if query_params:
            url += "?" + urlencode(query_params)

        session = await self._get_session()

        response = await session.request(
            method,
            url,
            headers=headers,
            data=body,
        )

        # Handle errors
        if response.status == 403:
            error_text = await response.text()
            raise AuthenticationError(f"S3 authentication failed: {error_text}")

        if response.status >= 400:
            error_text = await response.text()
            raise IntegrationError(
                f"S3 error ({response.status}): {error_text}"
            )

        return response

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Establish connection and verify credentials.

        Returns:
            True if connection successful
        """
        try:
            # Verify by listing buckets
            response = await self._request("GET", "/", bucket=None)
            await response.text()  # Consume response
            self._status = IntegrationStatus.CONNECTED
            logger.info(f"Connected to S3 for integration {self.integration_id}")
            return True

        except AuthenticationError:
            self._status = IntegrationStatus.AUTH_FAILED
            return False
        except Exception as e:
            self._status = IntegrationStatus.ERROR
            logger.error(f"Failed to connect to S3: {e}")
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect and clean up resources.

        Returns:
            True if disconnection successful
        """
        try:
            if self._session and not self._session.closed:
                await self._session.close()

            self._status = IntegrationStatus.DISCONNECTED
            logger.info(f"Disconnected from S3 for integration {self.integration_id}")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from S3: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the integration.

        Returns:
            Health check results with status and metrics
        """
        health = {
            "provider": self.PROVIDER_NAME,
            "status": "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "details": {},
        }

        try:
            start = datetime.utcnow()

            # Check bucket access if bucket is configured
            if self._bucket:
                response = await self._request(
                    "HEAD",
                    "/",
                    query_params={"list-type": "2", "max-keys": "1"},
                )
                await response.read()
            else:
                response = await self._request("GET", "/", bucket=None)
                await response.text()

            latency_ms = int((datetime.utcnow() - start).total_seconds() * 1000)

            health["status"] = "healthy"
            health["details"] = {
                "api_latency_ms": latency_ms,
                "bucket": self._bucket,
                "region": self._region,
                "credentials_valid": True,
            }

        except AuthenticationError:
            health["status"] = "unhealthy"
            health["details"]["error"] = "Authentication failed"
            health["details"]["credentials_valid"] = False

        except Exception as e:
            health["status"] = "unhealthy"
            health["details"]["error"] = str(e)

        return health

    # =========================================================================
    # Object Operations
    # =========================================================================

    async def list_objects(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
        limit: int = 1000,
        cursor: Optional[str] = None,
    ) -> Tuple[List[StorageObject], Optional[str]]:
        """
        List objects in a bucket.

        Args:
            prefix: Filter objects by prefix
            bucket: Bucket name
            limit: Maximum number of objects to return
            cursor: Continuation token

        Returns:
            Tuple of (list of StorageObject, next cursor)
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise IntegrationError("Bucket name required")

        query_params = {
            "list-type": "2",
            "max-keys": str(min(limit, 1000)),
        }

        if prefix:
            query_params["prefix"] = prefix
        if cursor:
            query_params["continuation-token"] = cursor

        response = await self._request("GET", "/", bucket=bucket, query_params=query_params)
        xml_content = await response.text()

        # Parse XML response
        objects = []
        next_cursor = None

        # Simple XML parsing (in production, use proper XML parser)
        import re

        # Extract objects
        for match in re.finditer(r"<Contents>(.*?)</Contents>", xml_content, re.DOTALL):
            content = match.group(1)

            key_match = re.search(r"<Key>(.*?)</Key>", content)
            size_match = re.search(r"<Size>(.*?)</Size>", content)
            modified_match = re.search(r"<LastModified>(.*?)</LastModified>", content)
            etag_match = re.search(r"<ETag>(.*?)</ETag>", content)
            storage_class_match = re.search(r"<StorageClass>(.*?)</StorageClass>", content)

            if key_match:
                key = key_match.group(1)
                objects.append(StorageObject(
                    key=key,
                    bucket=bucket,
                    size=int(size_match.group(1)) if size_match else 0,
                    content_type=mimetypes.guess_type(key)[0] or "application/octet-stream",
                    last_modified=datetime.fromisoformat(
                        modified_match.group(1).replace("Z", "+00:00")
                    ) if modified_match else None,
                    etag=etag_match.group(1).strip('"') if etag_match else None,
                    storage_class=storage_class_match.group(1) if storage_class_match else None,
                ))

        # Check for continuation
        truncated_match = re.search(r"<IsTruncated>(.*?)</IsTruncated>", xml_content)
        if truncated_match and truncated_match.group(1).lower() == "true":
            token_match = re.search(
                r"<NextContinuationToken>(.*?)</NextContinuationToken>",
                xml_content
            )
            if token_match:
                next_cursor = token_match.group(1)

        return objects, next_cursor

    async def get_object(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> Optional[StorageObject]:
        """
        Get object metadata.

        Args:
            key: Object key
            bucket: Bucket name

        Returns:
            StorageObject with metadata or None if not found
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise IntegrationError("Bucket name required")

        try:
            response = await self._request("HEAD", f"/{key}", bucket=bucket)

            return StorageObject(
                key=key,
                bucket=bucket,
                size=int(response.headers.get("Content-Length", 0)),
                content_type=response.headers.get("Content-Type", "application/octet-stream"),
                last_modified=datetime.strptime(
                    response.headers.get("Last-Modified", ""),
                    "%a, %d %b %Y %H:%M:%S %Z"
                ) if response.headers.get("Last-Modified") else None,
                etag=response.headers.get("ETag", "").strip('"'),
                storage_class=response.headers.get("x-amz-storage-class"),
                metadata={
                    k.replace("x-amz-meta-", ""): v
                    for k, v in response.headers.items()
                    if k.lower().startswith("x-amz-meta-")
                },
            )
        except IntegrationError as e:
            if "404" in str(e):
                return None
            raise

    async def download_object(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        """
        Download object content.

        Args:
            key: Object key
            bucket: Bucket name

        Returns:
            Object content as bytes
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise IntegrationError("Bucket name required")

        response = await self._request("GET", f"/{key}", bucket=bucket)
        return await response.read()

    async def download_object_stream(
        self,
        key: str,
        bucket: Optional[str] = None,
        chunk_size: int = 8192,
    ) -> AsyncIterator[bytes]:
        """
        Download object content as a stream.

        Args:
            key: Object key
            bucket: Bucket name
            chunk_size: Size of chunks to yield

        Yields:
            Chunks of object content
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise IntegrationError("Bucket name required")

        response = await self._request(
            "GET",
            f"/{key}",
            bucket=bucket,
            stream_response=True,
        )

        async for chunk in response.content.iter_chunked(chunk_size):
            yield chunk

    async def upload_object(
        self,
        key: str,
        data: bytes,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        storage_class: Optional[str] = None,
    ) -> StorageObject:
        """
        Upload an object to S3.

        Args:
            key: Object key
            data: Object content
            bucket: Bucket name
            content_type: Content type (auto-detected if not provided)
            metadata: Custom metadata
            storage_class: Storage class

        Returns:
            Created StorageObject
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise IntegrationError("Bucket name required")

        # Use multipart upload for large files
        if len(data) > self.MULTIPART_THRESHOLD:
            return await self._multipart_upload(
                key, data, bucket, content_type, metadata, storage_class
            )

        # Single PUT request for smaller files
        headers = {}

        if content_type:
            headers["Content-Type"] = content_type
        else:
            guessed_type = mimetypes.guess_type(key)[0]
            headers["Content-Type"] = guessed_type or "application/octet-stream"

        if metadata:
            for k, v in metadata.items():
                headers[f"x-amz-meta-{k}"] = v

        storage_class = storage_class or self._storage_class
        if storage_class:
            headers["x-amz-storage-class"] = storage_class

        # Add encryption headers
        if self._encryption == "AES256":
            headers["x-amz-server-side-encryption"] = "AES256"
        elif self._encryption == "aws:kms":
            headers["x-amz-server-side-encryption"] = "aws:kms"
            if self._kms_key_id:
                headers["x-amz-server-side-encryption-aws-kms-key-id"] = self._kms_key_id

        response = await self._request(
            "PUT",
            f"/{key}",
            bucket=bucket,
            headers=headers,
            body=data,
        )

        etag = response.headers.get("ETag", "").strip('"')

        return StorageObject(
            key=key,
            bucket=bucket,
            size=len(data),
            content_type=headers.get("Content-Type", "application/octet-stream"),
            last_modified=datetime.utcnow(),
            etag=etag,
            storage_class=storage_class,
            metadata=metadata,
        )

    async def _multipart_upload(
        self,
        key: str,
        data: bytes,
        bucket: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        storage_class: Optional[str] = None,
    ) -> StorageObject:
        """
        Perform multipart upload for large files.

        Args:
            key: Object key
            data: Object content
            bucket: Bucket name
            content_type: Content type
            metadata: Custom metadata
            storage_class: Storage class

        Returns:
            Created StorageObject
        """
        # Initiate multipart upload
        headers = {}

        if content_type:
            headers["Content-Type"] = content_type
        else:
            guessed_type = mimetypes.guess_type(key)[0]
            headers["Content-Type"] = guessed_type or "application/octet-stream"

        if metadata:
            for k, v in metadata.items():
                headers[f"x-amz-meta-{k}"] = v

        storage_class = storage_class or self._storage_class
        if storage_class:
            headers["x-amz-storage-class"] = storage_class

        if self._encryption == "AES256":
            headers["x-amz-server-side-encryption"] = "AES256"
        elif self._encryption == "aws:kms":
            headers["x-amz-server-side-encryption"] = "aws:kms"
            if self._kms_key_id:
                headers["x-amz-server-side-encryption-aws-kms-key-id"] = self._kms_key_id

        response = await self._request(
            "POST",
            f"/{key}",
            bucket=bucket,
            query_params={"uploads": ""},
            headers=headers,
        )

        xml_content = await response.text()

        # Extract upload ID
        import re
        upload_id_match = re.search(r"<UploadId>(.*?)</UploadId>", xml_content)
        if not upload_id_match:
            raise IntegrationError("Failed to initiate multipart upload")

        upload_id = upload_id_match.group(1)

        try:
            # Upload parts
            parts = []
            part_number = 1
            offset = 0

            while offset < len(data):
                chunk = data[offset:offset + self.MULTIPART_CHUNK_SIZE]

                part_response = await self._request(
                    "PUT",
                    f"/{key}",
                    bucket=bucket,
                    query_params={
                        "partNumber": str(part_number),
                        "uploadId": upload_id,
                    },
                    body=chunk,
                )

                etag = part_response.headers.get("ETag", "").strip('"')
                parts.append({
                    "PartNumber": part_number,
                    "ETag": etag,
                })

                part_number += 1
                offset += self.MULTIPART_CHUNK_SIZE

            # Complete multipart upload
            parts_xml = "".join(
                f"<Part><PartNumber>{p['PartNumber']}</PartNumber><ETag>{p['ETag']}</ETag></Part>"
                for p in parts
            )
            complete_xml = f"<CompleteMultipartUpload>{parts_xml}</CompleteMultipartUpload>"

            complete_response = await self._request(
                "POST",
                f"/{key}",
                bucket=bucket,
                query_params={"uploadId": upload_id},
                headers={"Content-Type": "application/xml"},
                body=complete_xml.encode("utf-8"),
            )

            complete_content = await complete_response.text()
            etag_match = re.search(r"<ETag>(.*?)</ETag>", complete_content)
            final_etag = etag_match.group(1).strip('"') if etag_match else ""

            return StorageObject(
                key=key,
                bucket=bucket,
                size=len(data),
                content_type=headers.get("Content-Type", "application/octet-stream"),
                last_modified=datetime.utcnow(),
                etag=final_etag,
                storage_class=storage_class,
                metadata=metadata,
            )

        except Exception:
            # Abort multipart upload on error
            try:
                await self._request(
                    "DELETE",
                    f"/{key}",
                    bucket=bucket,
                    query_params={"uploadId": upload_id},
                )
            except Exception:
                pass
            raise

    async def delete_object(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """
        Delete an object.

        Args:
            key: Object key
            bucket: Bucket name

        Returns:
            True if deletion successful
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise IntegrationError("Bucket name required")

        await self._request("DELETE", f"/{key}", bucket=bucket)
        return True

    async def delete_objects(
        self,
        keys: List[str],
        bucket: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Delete multiple objects.

        Args:
            keys: List of object keys
            bucket: Bucket name

        Returns:
            Dictionary mapping keys to deletion success
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise IntegrationError("Bucket name required")

        # Build delete request XML
        objects_xml = "".join(f"<Object><Key>{key}</Key></Object>" for key in keys)
        delete_xml = f'<?xml version="1.0" encoding="UTF-8"?><Delete>{objects_xml}</Delete>'

        # Calculate MD5 for Content-MD5 header
        content_md5 = base64.b64encode(
            hashlib.md5(delete_xml.encode("utf-8")).digest()
        ).decode("utf-8")

        response = await self._request(
            "POST",
            "/",
            bucket=bucket,
            query_params={"delete": ""},
            headers={
                "Content-Type": "application/xml",
                "Content-MD5": content_md5,
            },
            body=delete_xml.encode("utf-8"),
        )

        xml_content = await response.text()

        # Parse response
        import re
        results = {key: True for key in keys}

        for match in re.finditer(r"<Error>.*?<Key>(.*?)</Key>.*?</Error>", xml_content, re.DOTALL):
            failed_key = match.group(1)
            results[failed_key] = False

        return results

    async def copy_object(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
        dest_bucket: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """
        Copy an object.

        Args:
            source_key: Source object key
            dest_key: Destination object key
            source_bucket: Source bucket
            dest_bucket: Destination bucket
            metadata: New metadata (replaces existing if provided)

        Returns:
            Created StorageObject
        """
        source_bucket = source_bucket or self._bucket
        dest_bucket = dest_bucket or self._bucket

        if not source_bucket or not dest_bucket:
            raise IntegrationError("Bucket names required")

        headers = {
            "x-amz-copy-source": f"/{source_bucket}/{source_key}",
        }

        if metadata:
            headers["x-amz-metadata-directive"] = "REPLACE"
            for k, v in metadata.items():
                headers[f"x-amz-meta-{k}"] = v
        else:
            headers["x-amz-metadata-directive"] = "COPY"

        response = await self._request(
            "PUT",
            f"/{dest_key}",
            bucket=dest_bucket,
            headers=headers,
        )

        xml_content = await response.text()

        import re
        etag_match = re.search(r"<ETag>(.*?)</ETag>", xml_content)
        etag = etag_match.group(1).strip('"') if etag_match else ""

        # Get object info
        return await self.get_object(dest_key, dest_bucket)

    # =========================================================================
    # Presigned URLs
    # =========================================================================

    def generate_presigned_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in: int = DEFAULT_PRESIGNED_EXPIRY,
        method: str = "GET",
        content_type: Optional[str] = None,
    ) -> str:
        """
        Generate a presigned URL for temporary access.

        Args:
            key: Object key
            bucket: Bucket name
            expires_in: URL validity in seconds
            method: HTTP method (GET, PUT)
            content_type: Content type for PUT requests

        Returns:
            Presigned URL
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise IntegrationError("Bucket name required")

        expires_in = min(expires_in, self.MAX_PRESIGNED_EXPIRY)

        # Current time
        now = datetime.utcnow()
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")

        # Build query parameters
        credential_scope = f"{date_stamp}/{self._region}/{self.SERVICE}/aws4_request"

        query_params = {
            "X-Amz-Algorithm": self.ALGORITHM,
            "X-Amz-Credential": f"{self._access_key_id}/{credential_scope}",
            "X-Amz-Date": amz_date,
            "X-Amz-Expires": str(expires_in),
            "X-Amz-SignedHeaders": "host",
        }

        if self._session_token:
            query_params["X-Amz-Security-Token"] = self._session_token

        # Headers to sign
        host = self._get_host(bucket)
        headers_to_sign = {"host": host}

        # Build path
        if self._path_style:
            path = f"/{bucket}/{key}"
        else:
            path = f"/{key}"

        # Create canonical request
        canonical_request = self._create_canonical_request(
            method,
            path,
            query_params,
            headers_to_sign,
            "UNSIGNED-PAYLOAD",
        )

        # Create string to sign
        canonical_request_hash = hashlib.sha256(
            canonical_request.encode("utf-8")
        ).hexdigest()
        string_to_sign = self._create_string_to_sign(
            amz_date, date_stamp, canonical_request_hash
        )

        # Calculate signature
        signing_key = self._get_signature_key(date_stamp)
        signature = hmac.new(
            signing_key,
            string_to_sign.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        query_params["X-Amz-Signature"] = signature

        # Build URL
        endpoint = self._get_endpoint(bucket)
        if self._path_style:
            url = f"{endpoint}/{bucket}/{key}"
        else:
            url = f"{endpoint}/{key}"

        return url + "?" + urlencode(query_params)

    async def generate_presigned_upload_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in: int = DEFAULT_PRESIGNED_EXPIRY,
        content_type: str = "application/octet-stream",
        max_size: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a presigned URL for uploading.

        Args:
            key: Object key
            bucket: Bucket name
            expires_in: URL validity in seconds
            content_type: Content type
            max_size: Maximum file size (for policy)
            metadata: Custom metadata to include

        Returns:
            Dictionary with upload URL and fields
        """
        url = self.generate_presigned_url(
            key=key,
            bucket=bucket,
            expires_in=expires_in,
            method="PUT",
            content_type=content_type,
        )

        return {
            "url": url,
            "method": "PUT",
            "headers": {
                "Content-Type": content_type,
            },
            "expires_at": (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat(),
        }

    # =========================================================================
    # Bucket Operations
    # =========================================================================

    async def list_buckets(self) -> List[Dict[str, Any]]:
        """
        List all buckets.

        Returns:
            List of bucket information
        """
        response = await self._request("GET", "/", bucket=None)
        xml_content = await response.text()

        import re
        buckets = []

        for match in re.finditer(r"<Bucket>(.*?)</Bucket>", xml_content, re.DOTALL):
            content = match.group(1)
            name_match = re.search(r"<Name>(.*?)</Name>", content)
            date_match = re.search(r"<CreationDate>(.*?)</CreationDate>", content)

            if name_match:
                buckets.append({
                    "name": name_match.group(1),
                    "creation_date": datetime.fromisoformat(
                        date_match.group(1).replace("Z", "+00:00")
                    ) if date_match else None,
                })

        return buckets

    async def create_bucket(
        self,
        bucket_name: str,
        region: Optional[str] = None,
    ) -> bool:
        """
        Create a new bucket.

        Args:
            bucket_name: Bucket name
            region: AWS region

        Returns:
            True if creation successful
        """
        region = region or self._region

        body = None
        if region != "us-east-1":
            body = f"""<?xml version="1.0" encoding="UTF-8"?>
<CreateBucketConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
    <LocationConstraint>{region}</LocationConstraint>
</CreateBucketConfiguration>""".encode("utf-8")

        await self._request(
            "PUT",
            "/",
            bucket=bucket_name,
            body=body,
        )

        return True

    async def delete_bucket(self, bucket_name: str) -> bool:
        """
        Delete a bucket (must be empty).

        Args:
            bucket_name: Bucket name

        Returns:
            True if deletion successful
        """
        await self._request("DELETE", "/", bucket=bucket_name)
        return True

    async def get_bucket_location(self, bucket_name: str) -> str:
        """
        Get bucket region/location.

        Args:
            bucket_name: Bucket name

        Returns:
            Region name
        """
        response = await self._request(
            "GET",
            "/",
            bucket=bucket_name,
            query_params={"location": ""},
        )

        xml_content = await response.text()

        import re
        match = re.search(
            r"<LocationConstraint>(.*?)</LocationConstraint>",
            xml_content
        )

        if match and match.group(1):
            return match.group(1)
        return "us-east-1"  # Default region

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def object_exists(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """
        Check if an object exists.

        Args:
            key: Object key
            bucket: Bucket name

        Returns:
            True if object exists
        """
        obj = await self.get_object(key, bucket)
        return obj is not None

    def get_public_url(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> str:
        """
        Get public URL for an object (bucket must have public access).

        Args:
            key: Object key
            bucket: Bucket name

        Returns:
            Public URL
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise IntegrationError("Bucket name required")

        if self._path_style:
            return f"https://s3.{self._region}.amazonaws.com/{bucket}/{key}"
        else:
            return f"https://{bucket}.s3.{self._region}.amazonaws.com/{key}"

    async def get_storage_stats(
        self,
        bucket: Optional[str] = None,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Get storage statistics for a bucket or prefix.

        Args:
            bucket: Bucket name
            prefix: Prefix to filter

        Returns:
            Storage statistics
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise IntegrationError("Bucket name required")

        total_size = 0
        object_count = 0
        cursor = None

        while True:
            objects, cursor = await self.list_objects(
                prefix=prefix,
                bucket=bucket,
                limit=1000,
                cursor=cursor,
            )

            for obj in objects:
                total_size += obj.size
                object_count += 1

            if not cursor:
                break

        return {
            "bucket": bucket,
            "prefix": prefix,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_size_gb": round(total_size / (1024 * 1024 * 1024), 4),
            "object_count": object_count,
        }

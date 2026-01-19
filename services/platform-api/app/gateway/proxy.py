"""
Proxy Handler

Request proxying and transformation:
- Forward requests to backend services
- Transform requests/responses
- Handle streaming
- Connection pooling
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import json
import re
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

logger = logging.getLogger(__name__)


class ProxyProtocol(str, Enum):
    """Supported proxy protocols."""
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "ws"
    WEBSOCKET_SECURE = "wss"
    GRPC = "grpc"


@dataclass
class ProxyRequest:
    """Proxy request representation."""
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    client_ip: str = ""
    request_id: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_type(self) -> str:
        """Get content type."""
        return self.headers.get("content-type", "application/octet-stream")

    @property
    def is_json(self) -> bool:
        """Check if JSON content."""
        return "application/json" in self.content_type

    def get_json(self) -> Optional[Dict[str, Any]]:
        """Parse body as JSON."""
        if self.body and self.is_json:
            try:
                return json.loads(self.body)
            except json.JSONDecodeError:
                return None
        return None


@dataclass
class ProxyResponse:
    """Proxy response representation."""
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    response_time_ms: float = 0.0
    backend_id: str = ""
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Check if successful response."""
        return 200 <= self.status_code < 400

    @property
    def is_error(self) -> bool:
        """Check if error response."""
        return self.status_code >= 400


@dataclass
class ProxyConfig:
    """Proxy configuration."""
    # Timeouts
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 30.0
    write_timeout_seconds: float = 30.0
    total_timeout_seconds: float = 60.0

    # Retries
    max_retries: int = 3
    retry_delay_seconds: float = 0.1
    retry_backoff_multiplier: float = 2.0
    retry_on_statuses: List[int] = field(default_factory=lambda: [502, 503, 504])

    # Connection pooling
    pool_size: int = 100
    pool_max_size: int = 200
    pool_timeout_seconds: float = 30.0
    keepalive_seconds: float = 30.0

    # Request limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_response_size: int = 50 * 1024 * 1024  # 50MB

    # Headers
    preserve_host: bool = False
    add_x_forwarded_headers: bool = True
    strip_headers: List[str] = field(default_factory=lambda: [
        "connection", "keep-alive", "transfer-encoding", "upgrade"
    ])

    # SSL
    verify_ssl: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "connect_timeout": self.connect_timeout_seconds,
            "read_timeout": self.read_timeout_seconds,
            "max_retries": self.max_retries,
            "pool_size": self.pool_size,
            "max_request_size": self.max_request_size,
        }


class RequestTransformer(ABC):
    """Abstract request transformer."""

    @abstractmethod
    async def transform(self, request: ProxyRequest) -> ProxyRequest:
        """Transform request before forwarding."""
        pass


class ResponseTransformer(ABC):
    """Abstract response transformer."""

    @abstractmethod
    async def transform(
        self,
        response: ProxyResponse,
        request: ProxyRequest,
    ) -> ProxyResponse:
        """Transform response before returning."""
        pass


class HeaderTransformer(RequestTransformer):
    """Transform request headers."""

    def __init__(
        self,
        add_headers: Optional[Dict[str, str]] = None,
        remove_headers: Optional[List[str]] = None,
        rename_headers: Optional[Dict[str, str]] = None,
    ):
        self.add_headers = add_headers or {}
        self.remove_headers = [h.lower() for h in (remove_headers or [])]
        self.rename_headers = {k.lower(): v for k, v in (rename_headers or {}).items()}

    async def transform(self, request: ProxyRequest) -> ProxyRequest:
        """Transform headers."""
        new_headers = {}

        for key, value in request.headers.items():
            lower_key = key.lower()

            # Skip removed headers
            if lower_key in self.remove_headers:
                continue

            # Rename if needed
            if lower_key in self.rename_headers:
                new_headers[self.rename_headers[lower_key]] = value
            else:
                new_headers[key] = value

        # Add new headers
        new_headers.update(self.add_headers)

        request.headers = new_headers
        return request


class PathTransformer(RequestTransformer):
    """Transform request path."""

    def __init__(
        self,
        strip_prefix: Optional[str] = None,
        add_prefix: Optional[str] = None,
        path_rewrites: Optional[Dict[str, str]] = None,
    ):
        self.strip_prefix = strip_prefix
        self.add_prefix = add_prefix
        self.path_rewrites = path_rewrites or {}
        self._compiled_rewrites = {
            re.compile(pattern): replacement
            for pattern, replacement in self.path_rewrites.items()
        }

    async def transform(self, request: ProxyRequest) -> ProxyRequest:
        """Transform path."""
        path = request.path

        # Strip prefix
        if self.strip_prefix and path.startswith(self.strip_prefix):
            path = path[len(self.strip_prefix):] or "/"

        # Apply rewrites
        for pattern, replacement in self._compiled_rewrites.items():
            path = pattern.sub(replacement, path)

        # Add prefix
        if self.add_prefix:
            path = self.add_prefix.rstrip("/") + "/" + path.lstrip("/")

        request.path = path
        return request


class BodyTransformer(RequestTransformer):
    """Transform request body."""

    def __init__(
        self,
        json_transforms: Optional[Dict[str, Callable[[Any], Any]]] = None,
        add_fields: Optional[Dict[str, Any]] = None,
        remove_fields: Optional[List[str]] = None,
    ):
        self.json_transforms = json_transforms or {}
        self.add_fields = add_fields or {}
        self.remove_fields = remove_fields or []

    async def transform(self, request: ProxyRequest) -> ProxyRequest:
        """Transform body."""
        if not request.is_json or not request.body:
            return request

        try:
            data = json.loads(request.body)

            # Apply transforms
            for field_path, transform in self.json_transforms.items():
                data = self._apply_transform(data, field_path, transform)

            # Remove fields
            for field_path in self.remove_fields:
                data = self._remove_field(data, field_path)

            # Add fields
            for field_path, value in self.add_fields.items():
                data = self._set_field(data, field_path, value)

            request.body = json.dumps(data).encode()

        except json.JSONDecodeError:
            pass

        return request

    def _apply_transform(
        self,
        data: Any,
        path: str,
        transform: Callable[[Any], Any],
    ) -> Any:
        """Apply transform at path."""
        parts = path.split(".")
        return self._navigate_and_apply(data, parts, transform)

    def _navigate_and_apply(
        self,
        data: Any,
        parts: List[str],
        transform: Callable[[Any], Any],
    ) -> Any:
        """Navigate to path and apply transform."""
        if not parts:
            return transform(data)

        if isinstance(data, dict):
            key = parts[0]
            if key in data:
                data[key] = self._navigate_and_apply(data[key], parts[1:], transform)

        return data

    def _remove_field(self, data: Any, path: str) -> Any:
        """Remove field at path."""
        parts = path.split(".")
        return self._navigate_and_remove(data, parts)

    def _navigate_and_remove(self, data: Any, parts: List[str]) -> Any:
        """Navigate to path and remove."""
        if len(parts) == 1 and isinstance(data, dict):
            data.pop(parts[0], None)
            return data

        if isinstance(data, dict) and parts[0] in data:
            data[parts[0]] = self._navigate_and_remove(data[parts[0]], parts[1:])

        return data

    def _set_field(self, data: Any, path: str, value: Any) -> Any:
        """Set field at path."""
        parts = path.split(".")
        return self._navigate_and_set(data, parts, value)

    def _navigate_and_set(self, data: Any, parts: List[str], value: Any) -> Any:
        """Navigate to path and set value."""
        if len(parts) == 1:
            if isinstance(data, dict):
                data[parts[0]] = value
            return data

        if not isinstance(data, dict):
            return data

        key = parts[0]
        if key not in data:
            data[key] = {}

        data[key] = self._navigate_and_set(data[key], parts[1:], value)
        return data


class ResponseHeaderTransformer(ResponseTransformer):
    """Transform response headers."""

    def __init__(
        self,
        add_headers: Optional[Dict[str, str]] = None,
        remove_headers: Optional[List[str]] = None,
    ):
        self.add_headers = add_headers or {}
        self.remove_headers = [h.lower() for h in (remove_headers or [])]

    async def transform(
        self,
        response: ProxyResponse,
        request: ProxyRequest,
    ) -> ProxyResponse:
        """Transform response headers."""
        new_headers = {
            k: v for k, v in response.headers.items()
            if k.lower() not in self.remove_headers
        }
        new_headers.update(self.add_headers)
        response.headers = new_headers
        return response


class ResponseBodyTransformer(ResponseTransformer):
    """Transform response body."""

    def __init__(
        self,
        json_transforms: Optional[Dict[str, Callable[[Any], Any]]] = None,
    ):
        self.json_transforms = json_transforms or {}

    async def transform(
        self,
        response: ProxyResponse,
        request: ProxyRequest,
    ) -> ProxyResponse:
        """Transform response body."""
        content_type = response.headers.get("content-type", "")

        if "application/json" not in content_type or not response.body:
            return response

        try:
            data = json.loads(response.body)

            for path, transform in self.json_transforms.items():
                data = self._apply_transform(data, path.split("."), transform)

            response.body = json.dumps(data).encode()

        except json.JSONDecodeError:
            pass

        return response

    def _apply_transform(
        self,
        data: Any,
        parts: List[str],
        transform: Callable[[Any], Any],
    ) -> Any:
        """Apply transform at path."""
        if not parts:
            return transform(data)

        if isinstance(data, dict):
            key = parts[0]
            if key in data:
                data[key] = self._apply_transform(data[key], parts[1:], transform)

        return data


class CompositeRequestTransformer(RequestTransformer):
    """Chain multiple request transformers."""

    def __init__(self, transformers: List[RequestTransformer]):
        self.transformers = transformers

    async def transform(self, request: ProxyRequest) -> ProxyRequest:
        """Apply all transformers in sequence."""
        for transformer in self.transformers:
            request = await transformer.transform(request)
        return request


class CompositeResponseTransformer(ResponseTransformer):
    """Chain multiple response transformers."""

    def __init__(self, transformers: List[ResponseTransformer]):
        self.transformers = transformers

    async def transform(
        self,
        response: ProxyResponse,
        request: ProxyRequest,
    ) -> ProxyResponse:
        """Apply all transformers in sequence."""
        for transformer in self.transformers:
            response = await transformer.transform(response, request)
        return response


class ConnectionPool:
    """HTTP connection pool."""

    def __init__(
        self,
        max_connections: int = 100,
        max_connections_per_host: int = 10,
        keepalive_timeout: float = 30.0,
    ):
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.keepalive_timeout = keepalive_timeout
        self._pools: Dict[str, asyncio.Queue] = {}
        self._active: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def get_connection(self, host: str, port: int, ssl: bool = False):
        """Get connection from pool."""
        key = f"{host}:{port}:{ssl}"

        async with self._lock:
            if key not in self._pools:
                self._pools[key] = asyncio.Queue()
                self._active[key] = 0

        # Try to get from pool
        try:
            conn = self._pools[key].get_nowait()
            return conn
        except asyncio.QueueEmpty:
            pass

        # Create new connection if under limit
        async with self._lock:
            if self._active[key] < self.max_connections_per_host:
                self._active[key] += 1
                # In production, create actual connection
                return {"host": host, "port": port, "ssl": ssl, "id": id}

        # Wait for available connection
        return await asyncio.wait_for(
            self._pools[key].get(),
            timeout=self.keepalive_timeout,
        )

    async def release_connection(self, host: str, port: int, conn: Any, ssl: bool = False):
        """Return connection to pool."""
        key = f"{host}:{port}:{ssl}"

        if key in self._pools:
            try:
                self._pools[key].put_nowait(conn)
            except asyncio.QueueFull:
                async with self._lock:
                    self._active[key] -= 1

    async def close_all(self):
        """Close all connections."""
        self._pools.clear()
        self._active.clear()


class ProxyHandler:
    """
    HTTP proxy handler.

    Forwards requests to backend services with:
    - Connection pooling
    - Request/response transformation
    - Retry logic
    - Timeout handling
    """

    def __init__(
        self,
        config: Optional[ProxyConfig] = None,
    ):
        self.config = config or ProxyConfig()
        self._pool = ConnectionPool(
            max_connections=self.config.pool_size,
            max_connections_per_host=self.config.pool_size // 10,
            keepalive_timeout=self.config.keepalive_seconds,
        )
        self._request_transformers: List[RequestTransformer] = []
        self._response_transformers: List[ResponseTransformer] = []
        self._request_count = 0
        self._error_count = 0

    def add_request_transformer(self, transformer: RequestTransformer) -> None:
        """Add request transformer."""
        self._request_transformers.append(transformer)

    def add_response_transformer(self, transformer: ResponseTransformer) -> None:
        """Add response transformer."""
        self._response_transformers.append(transformer)

    async def proxy(
        self,
        request: ProxyRequest,
        target_url: str,
        backend_id: str = "",
    ) -> ProxyResponse:
        """Proxy request to target."""
        self._request_count += 1
        start_time = datetime.utcnow()

        # Transform request
        for transformer in self._request_transformers:
            request = await transformer.transform(request)

        # Parse target
        parsed = urlparse(target_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        ssl = parsed.scheme == "https"

        # Build full URL
        path = request.path
        if request.query_params:
            path = f"{path}?{urlencode(request.query_params)}"
        full_url = urljoin(target_url, path)

        # Add X-Forwarded headers
        if self.config.add_x_forwarded_headers:
            request.headers["X-Forwarded-For"] = request.client_ip
            request.headers["X-Forwarded-Proto"] = "https" if ssl else "http"
            request.headers["X-Forwarded-Host"] = request.headers.get("host", host)
            request.headers["X-Request-ID"] = request.request_id

        # Preserve or replace host
        if not self.config.preserve_host:
            request.headers["host"] = f"{host}:{port}" if port not in (80, 443) else host

        # Strip hop-by-hop headers
        for header in self.config.strip_headers:
            request.headers.pop(header, None)
            request.headers.pop(header.title(), None)

        # Execute with retries
        response = await self._execute_with_retry(
            request, host, port, ssl, full_url, backend_id
        )

        # Calculate response time
        response.response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Transform response
        for transformer in self._response_transformers:
            response = await transformer.transform(response, request)

        return response

    async def _execute_with_retry(
        self,
        request: ProxyRequest,
        host: str,
        port: int,
        ssl: bool,
        url: str,
        backend_id: str,
    ) -> ProxyResponse:
        """Execute request with retry logic."""
        last_error = None
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._execute_request(
                    request, host, port, ssl, url, backend_id
                )

                # Check if should retry
                if response.status_code in self.config.retry_on_statuses:
                    if attempt < self.config.max_retries:
                        await asyncio.sleep(delay)
                        delay *= self.config.retry_backoff_multiplier
                        continue

                return response

            except asyncio.TimeoutError as e:
                last_error = f"Timeout: {e}"
                self._error_count += 1
                if attempt < self.config.max_retries:
                    await asyncio.sleep(delay)
                    delay *= self.config.retry_backoff_multiplier

            except Exception as e:
                last_error = str(e)
                self._error_count += 1
                if attempt < self.config.max_retries:
                    await asyncio.sleep(delay)
                    delay *= self.config.retry_backoff_multiplier

        # Return error response
        return ProxyResponse(
            status_code=502,
            headers={"content-type": "application/json"},
            body=json.dumps({"error": "Bad Gateway", "detail": last_error}).encode(),
            backend_id=backend_id,
            error=last_error,
        )

    async def _execute_request(
        self,
        request: ProxyRequest,
        host: str,
        port: int,
        ssl: bool,
        url: str,
        backend_id: str,
    ) -> ProxyResponse:
        """Execute single HTTP request."""
        # In production, use aiohttp or httpx
        # This is a simulation for the framework

        try:
            # Get connection from pool
            conn = await self._pool.get_connection(host, port, ssl)

            try:
                # Simulate request execution
                # async with aiohttp.ClientSession() as session:
                #     async with session.request(
                #         method=request.method,
                #         url=url,
                #         headers=request.headers,
                #         data=request.body,
                #         timeout=aiohttp.ClientTimeout(total=self.config.total_timeout_seconds),
                #     ) as resp:
                #         body = await resp.read()
                #         return ProxyResponse(
                #             status_code=resp.status,
                #             headers=dict(resp.headers),
                #             body=body,
                #             backend_id=backend_id,
                #         )

                # Simulation response
                return ProxyResponse(
                    status_code=200,
                    headers={"content-type": "application/json"},
                    body=json.dumps({"proxied": True, "url": url}).encode(),
                    backend_id=backend_id,
                )

            finally:
                # Return connection to pool
                await self._pool.release_connection(host, port, conn, ssl)

        except Exception as e:
            logger.error(f"Proxy error to {url}: {e}")
            raise

    async def close(self) -> None:
        """Close proxy handler."""
        await self._pool.close_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get proxy statistics."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
        }


class StreamingProxyHandler:
    """
    Streaming proxy handler.

    Supports chunked transfer and WebSocket proxying.
    """

    def __init__(self, config: Optional[ProxyConfig] = None):
        self.config = config or ProxyConfig()
        self._active_streams: Dict[str, Any] = {}

    async def proxy_stream(
        self,
        request: ProxyRequest,
        target_url: str,
        on_chunk: Callable[[bytes], Awaitable[None]],
    ) -> ProxyResponse:
        """Proxy streaming request."""
        stream_id = request.request_id or str(id(request))
        self._active_streams[stream_id] = {"started": datetime.utcnow()}

        try:
            # In production:
            # async with aiohttp.ClientSession() as session:
            #     async with session.request(
            #         method=request.method,
            #         url=target_url,
            #         headers=request.headers,
            #         data=request.body,
            #     ) as resp:
            #         async for chunk in resp.content.iter_any():
            #             await on_chunk(chunk)
            #         return ProxyResponse(status_code=resp.status, headers=dict(resp.headers))

            # Simulation
            await on_chunk(b"chunk1")
            await on_chunk(b"chunk2")

            return ProxyResponse(
                status_code=200,
                headers={"content-type": "text/event-stream"},
            )

        finally:
            del self._active_streams[stream_id]

    async def proxy_websocket(
        self,
        request: ProxyRequest,
        target_url: str,
        client_ws: Any,
    ) -> None:
        """Proxy WebSocket connection."""
        stream_id = request.request_id or str(id(request))
        self._active_streams[stream_id] = {"type": "websocket", "started": datetime.utcnow()}

        try:
            # In production: establish WebSocket to backend and forward messages
            # async with websockets.connect(target_url) as backend_ws:
            #     await asyncio.gather(
            #         self._forward_to_backend(client_ws, backend_ws),
            #         self._forward_to_client(backend_ws, client_ws),
            #     )
            pass

        finally:
            del self._active_streams[stream_id]

    def get_active_streams(self) -> int:
        """Get number of active streams."""
        return len(self._active_streams)


class GRPCProxyHandler:
    """
    gRPC proxy handler.

    Supports unary and streaming gRPC calls.
    """

    def __init__(
        self,
        config: Optional[ProxyConfig] = None,
    ):
        self.config = config or ProxyConfig()
        self._channels: Dict[str, Any] = {}

    async def proxy_unary(
        self,
        service: str,
        method: str,
        request_data: bytes,
        target: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> tuple:
        """Proxy unary gRPC call."""
        # In production, use grpcio-tools
        # channel = grpc.aio.insecure_channel(target)
        # stub = create_stub(channel, service)
        # response = await getattr(stub, method)(request_data, metadata=metadata)
        # return response.SerializeToString(), {}

        return b"grpc_response", {}

    async def proxy_server_stream(
        self,
        service: str,
        method: str,
        request_data: bytes,
        target: str,
        on_response: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Proxy server-streaming gRPC call."""
        # In production:
        # async for response in stub.method(request):
        #     await on_response(response.SerializeToString())
        await on_response(b"grpc_stream_1")
        await on_response(b"grpc_stream_2")

    async def close(self) -> None:
        """Close all channels."""
        for channel in self._channels.values():
            # await channel.close()
            pass
        self._channels.clear()


class ProxyMetrics:
    """Proxy metrics collector."""

    def __init__(self):
        self._requests: Dict[str, int] = {}
        self._errors: Dict[str, int] = {}
        self._latencies: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def record_request(
        self,
        backend_id: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Record request metrics."""
        async with self._lock:
            self._requests[backend_id] = self._requests.get(backend_id, 0) + 1

            if not success:
                self._errors[backend_id] = self._errors.get(backend_id, 0) + 1

            if backend_id not in self._latencies:
                self._latencies[backend_id] = []
            self._latencies[backend_id].append(latency_ms)

            # Keep last 1000 latencies
            if len(self._latencies[backend_id]) > 1000:
                self._latencies[backend_id] = self._latencies[backend_id][-1000:]

    def get_metrics(self, backend_id: str) -> Dict[str, Any]:
        """Get metrics for backend."""
        requests = self._requests.get(backend_id, 0)
        errors = self._errors.get(backend_id, 0)
        latencies = self._latencies.get(backend_id, [])

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

        return {
            "requests": requests,
            "errors": errors,
            "error_rate": errors / max(1, requests),
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
        }

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all backends."""
        return {
            backend_id: self.get_metrics(backend_id)
            for backend_id in self._requests.keys()
        }


class ProxyManager:
    """
    Manages proxy handlers and routing.
    """

    def __init__(self):
        self._handlers: Dict[str, ProxyHandler] = {}
        self._streaming_handlers: Dict[str, StreamingProxyHandler] = {}
        self._metrics = ProxyMetrics()
        self._default_config = ProxyConfig()

    def create_handler(
        self,
        name: str,
        config: Optional[ProxyConfig] = None,
    ) -> ProxyHandler:
        """Create named proxy handler."""
        handler = ProxyHandler(config or self._default_config)
        self._handlers[name] = handler
        return handler

    def get_handler(self, name: str) -> Optional[ProxyHandler]:
        """Get proxy handler by name."""
        return self._handlers.get(name)

    def create_streaming_handler(
        self,
        name: str,
        config: Optional[ProxyConfig] = None,
    ) -> StreamingProxyHandler:
        """Create streaming proxy handler."""
        handler = StreamingProxyHandler(config or self._default_config)
        self._streaming_handlers[name] = handler
        return handler

    async def proxy(
        self,
        handler_name: str,
        request: ProxyRequest,
        target_url: str,
        backend_id: str = "",
    ) -> ProxyResponse:
        """Proxy using named handler."""
        handler = self._handlers.get(handler_name)
        if not handler:
            return ProxyResponse(
                status_code=500,
                error=f"Handler not found: {handler_name}",
            )

        response = await handler.proxy(request, target_url, backend_id)

        # Record metrics
        await self._metrics.record_request(
            backend_id or handler_name,
            response.is_success,
            response.response_time_ms,
        )

        return response

    def get_metrics(self) -> Dict[str, Any]:
        """Get all proxy metrics."""
        return {
            "handlers": list(self._handlers.keys()),
            "streaming_handlers": list(self._streaming_handlers.keys()),
            "backend_metrics": self._metrics.get_all_metrics(),
        }

    async def close_all(self) -> None:
        """Close all handlers."""
        for handler in self._handlers.values():
            await handler.close()
        self._handlers.clear()
        self._streaming_handlers.clear()

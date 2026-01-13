"""Load test scenarios for Builder Engine platform."""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4
import logging

import aiohttp
import websockets

from .base import LoadTestBase, LoadTestConfig, RequestResult, VirtualUser

logger = logging.getLogger(__name__)


class APILoadTest(LoadTestBase):
    """
    Load test for REST API endpoints.

    Tests typical API operations:
    - Agent CRUD
    - Call management
    - Analytics queries
    - Configuration retrieval
    """

    def __init__(self, config: LoadTestConfig):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.endpoints: List[Dict[str, Any]] = []

    async def setup(self) -> None:
        """Setup HTTP session and define endpoints."""
        timeout = aiohttp.ClientTimeout(
            total=self.config.request_timeout,
            connect=self.config.connection_timeout,
        )

        self.session = aiohttp.ClientSession(
            base_url=self.config.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "X-Organization-ID": self.config.organization_id,
            },
        )

        # Define endpoints to test with their weights
        self.endpoints = [
            {"method": "GET", "path": "/api/v1/agents", "weight": 20},
            {"method": "GET", "path": "/api/v1/agents/{agent_id}", "weight": 15},
            {"method": "GET", "path": "/api/v1/calls", "weight": 20},
            {"method": "GET", "path": "/api/v1/calls/{call_id}", "weight": 10},
            {"method": "GET", "path": "/api/v1/analytics/overview", "weight": 10},
            {"method": "GET", "path": "/api/v1/phone-numbers", "weight": 10},
            {"method": "GET", "path": "/api/v1/health", "weight": 15},
        ]

    async def teardown(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()

    async def run_iteration(self, user_id: int, iteration: int) -> RequestResult:
        """Execute a single API request."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        # Select endpoint based on weights
        endpoint = self._select_endpoint()
        path = self._resolve_path(endpoint["path"])
        method = endpoint["method"]

        request_id = f"{user_id}-{iteration}"
        start_time = time.time()

        try:
            async with self.session.request(method, path) as response:
                response_time = (time.time() - start_time) * 1000
                body = await response.read()

                return RequestResult(
                    request_id=request_id,
                    endpoint=path,
                    method=method,
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=200 <= response.status < 400,
                    response_size_bytes=len(body),
                )

        except aiohttp.ClientError as e:
            response_time = (time.time() - start_time) * 1000
            return RequestResult(
                request_id=request_id,
                endpoint=path,
                method=method,
                status_code=0,
                response_time_ms=response_time,
                success=False,
                error_message=str(e),
            )

    def _select_endpoint(self) -> Dict[str, Any]:
        """Select endpoint based on weights."""
        total_weight = sum(e["weight"] for e in self.endpoints)
        r = random.random() * total_weight

        cumulative = 0
        for endpoint in self.endpoints:
            cumulative += endpoint["weight"]
            if r <= cumulative:
                return endpoint

        return self.endpoints[-1]

    def _resolve_path(self, path: str) -> str:
        """Resolve path placeholders."""
        # Generate fake IDs for parameterized endpoints
        if "{agent_id}" in path:
            path = path.replace("{agent_id}", f"agent_{uuid4().hex[:12]}")
        if "{call_id}" in path:
            path = path.replace("{call_id}", f"call_{uuid4().hex[:12]}")
        return path


class CallSimulationTest(LoadTestBase):
    """
    Load test simulating concurrent voice calls.

    Tests the call lifecycle:
    1. Initiate call
    2. Send audio chunks
    3. Receive responses
    4. End call
    """

    def __init__(self, config: LoadTestConfig):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_calls: Dict[str, Dict[str, Any]] = {}

    async def setup(self) -> None:
        """Setup for call simulation."""
        timeout = aiohttp.ClientTimeout(
            total=self.config.request_timeout,
            connect=self.config.connection_timeout,
        )

        self.session = aiohttp.ClientSession(
            base_url=self.config.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def teardown(self) -> None:
        """End any active calls and cleanup."""
        # End any remaining calls
        for call_id in list(self.active_calls.keys()):
            try:
                await self._end_call(call_id)
            except Exception:
                pass

        if self.session:
            await self.session.close()

    async def run_iteration(self, user_id: int, iteration: int) -> RequestResult:
        """Run a single call simulation iteration."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        request_id = f"call-{user_id}-{iteration}"
        start_time = time.time()

        try:
            # Determine action based on iteration
            action = iteration % 4

            if action == 0:
                # Initiate new call
                result = await self._initiate_call(user_id, request_id)
            elif action == 1:
                # Send audio
                result = await self._send_audio_chunk(user_id, request_id)
            elif action == 2:
                # Get call status
                result = await self._get_call_status(user_id, request_id)
            else:
                # End call
                result = await self._end_call_for_user(user_id, request_id)

            return result

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint="call_simulation",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def _initiate_call(self, user_id: int, request_id: str) -> RequestResult:
        """Initiate a new call."""
        start_time = time.time()

        payload = {
            "agent_id": f"agent_{uuid4().hex[:12]}",
            "phone_number": f"+1555{random.randint(1000000, 9999999)}",
            "direction": "outbound",
            "metadata": {"user_id": user_id, "test": True},
        }

        async with self.session.post("/api/v1/calls", json=payload) as response:
            response_time = (time.time() - start_time) * 1000
            body = await response.json() if response.status == 200 else {}

            if response.status == 200:
                call_id = body.get("id", "")
                self.active_calls[f"user-{user_id}"] = {
                    "call_id": call_id,
                    "started_at": time.time(),
                }

            return RequestResult(
                request_id=request_id,
                endpoint="/api/v1/calls",
                method="POST",
                status_code=response.status,
                response_time_ms=response_time,
                success=response.status == 200,
                response_size_bytes=len(json.dumps(body)),
            )

    async def _send_audio_chunk(self, user_id: int, request_id: str) -> RequestResult:
        """Send audio chunk to active call."""
        start_time = time.time()

        call_data = self.active_calls.get(f"user-{user_id}")
        if not call_data:
            # No active call, initiate one
            return await self._initiate_call(user_id, request_id)

        call_id = call_data["call_id"]

        # Simulate audio chunk (base64 encoded)
        audio_chunk = "SGVsbG8gV29ybGQ=" * 100  # ~1.3KB of fake audio

        payload = {
            "audio": audio_chunk,
            "format": "pcm_16000",
        }

        async with self.session.post(
            f"/api/v1/calls/{call_id}/audio",
            json=payload,
        ) as response:
            response_time = (time.time() - start_time) * 1000

            return RequestResult(
                request_id=request_id,
                endpoint=f"/api/v1/calls/{call_id}/audio",
                method="POST",
                status_code=response.status,
                response_time_ms=response_time,
                success=response.status in (200, 202),
            )

    async def _get_call_status(self, user_id: int, request_id: str) -> RequestResult:
        """Get status of active call."""
        start_time = time.time()

        call_data = self.active_calls.get(f"user-{user_id}")
        if not call_data:
            return await self._initiate_call(user_id, request_id)

        call_id = call_data["call_id"]

        async with self.session.get(f"/api/v1/calls/{call_id}") as response:
            response_time = (time.time() - start_time) * 1000
            body = await response.read()

            return RequestResult(
                request_id=request_id,
                endpoint=f"/api/v1/calls/{call_id}",
                method="GET",
                status_code=response.status,
                response_time_ms=response_time,
                success=response.status == 200,
                response_size_bytes=len(body),
            )

    async def _end_call_for_user(self, user_id: int, request_id: str) -> RequestResult:
        """End call for a user."""
        call_data = self.active_calls.pop(f"user-{user_id}", None)
        if not call_data:
            return await self._initiate_call(user_id, request_id)

        return await self._end_call(call_data["call_id"], request_id)

    async def _end_call(
        self,
        call_id: str,
        request_id: str = "",
    ) -> RequestResult:
        """End a specific call."""
        start_time = time.time()

        async with self.session.post(f"/api/v1/calls/{call_id}/end") as response:
            response_time = (time.time() - start_time) * 1000

            return RequestResult(
                request_id=request_id or call_id,
                endpoint=f"/api/v1/calls/{call_id}/end",
                method="POST",
                status_code=response.status,
                response_time_ms=response_time,
                success=response.status in (200, 202, 204),
            )


class WebSocketLoadTest(LoadTestBase):
    """
    Load test for WebSocket connections.

    Tests:
    - Connection establishment
    - Message throughput
    - Connection stability
    - Concurrent connections
    """

    def __init__(self, config: LoadTestConfig):
        super().__init__(config)
        self.connections: Dict[int, websockets.WebSocketClientProtocol] = {}
        self.message_count: Dict[int, int] = {}

    async def setup(self) -> None:
        """Setup WebSocket connections."""
        pass  # Connections created per user

    async def teardown(self) -> None:
        """Close all WebSocket connections."""
        for ws in self.connections.values():
            try:
                await ws.close()
            except Exception:
                pass
        self.connections.clear()

    async def run_iteration(self, user_id: int, iteration: int) -> RequestResult:
        """Run a WebSocket test iteration."""
        request_id = f"ws-{user_id}-{iteration}"
        start_time = time.time()

        try:
            # Get or create connection
            ws = self.connections.get(user_id)

            if not ws or ws.closed:
                # Create new connection
                return await self._connect(user_id, request_id)

            # Alternate between send and receive
            if iteration % 2 == 0:
                return await self._send_message(user_id, ws, request_id)
            else:
                return await self._receive_message(user_id, ws, request_id)

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint="websocket",
                method="WS",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def _connect(self, user_id: int, request_id: str) -> RequestResult:
        """Establish WebSocket connection."""
        start_time = time.time()

        ws_url = self.config.websocket_url or f"{self.config.base_url.replace('http', 'ws')}/ws"

        try:
            ws = await asyncio.wait_for(
                websockets.connect(
                    ws_url,
                    extra_headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                    },
                ),
                timeout=self.config.connection_timeout,
            )

            self.connections[user_id] = ws
            self.message_count[user_id] = 0

            response_time = (time.time() - start_time) * 1000

            return RequestResult(
                request_id=request_id,
                endpoint="websocket/connect",
                method="WS",
                status_code=101,  # WebSocket upgrade
                response_time_ms=response_time,
                success=True,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return RequestResult(
                request_id=request_id,
                endpoint="websocket/connect",
                method="WS",
                status_code=0,
                response_time_ms=response_time,
                success=False,
                error_message=str(e),
            )

    async def _send_message(
        self,
        user_id: int,
        ws: websockets.WebSocketClientProtocol,
        request_id: str,
    ) -> RequestResult:
        """Send a message over WebSocket."""
        start_time = time.time()

        try:
            message = json.dumps({
                "type": "ping",
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "sequence": self.message_count.get(user_id, 0),
            })

            await asyncio.wait_for(ws.send(message), timeout=5.0)

            self.message_count[user_id] = self.message_count.get(user_id, 0) + 1

            response_time = (time.time() - start_time) * 1000

            return RequestResult(
                request_id=request_id,
                endpoint="websocket/send",
                method="WS",
                status_code=200,
                response_time_ms=response_time,
                success=True,
                response_size_bytes=len(message),
            )

        except Exception as e:
            # Connection may be dead, remove it
            self.connections.pop(user_id, None)

            return RequestResult(
                request_id=request_id,
                endpoint="websocket/send",
                method="WS",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def _receive_message(
        self,
        user_id: int,
        ws: websockets.WebSocketClientProtocol,
        request_id: str,
    ) -> RequestResult:
        """Receive a message from WebSocket."""
        start_time = time.time()

        try:
            message = await asyncio.wait_for(ws.recv(), timeout=5.0)

            response_time = (time.time() - start_time) * 1000

            return RequestResult(
                request_id=request_id,
                endpoint="websocket/receive",
                method="WS",
                status_code=200,
                response_time_ms=response_time,
                success=True,
                response_size_bytes=len(message) if isinstance(message, (str, bytes)) else 0,
            )

        except asyncio.TimeoutError:
            # Timeout is acceptable for receive (no message available)
            return RequestResult(
                request_id=request_id,
                endpoint="websocket/receive",
                method="WS",
                status_code=204,  # No content
                response_time_ms=(time.time() - start_time) * 1000,
                success=True,
            )

        except Exception as e:
            self.connections.pop(user_id, None)

            return RequestResult(
                request_id=request_id,
                endpoint="websocket/receive",
                method="WS",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )


class ConcurrentCallsTest(LoadTestBase):
    """
    Test maximum concurrent call capacity.

    Progressively increases concurrent calls until
    failure threshold is reached.
    """

    def __init__(self, config: LoadTestConfig):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_call_ids: List[str] = []
        self.max_concurrent_calls: int = 0
        self.target_concurrent: int = 10

    async def setup(self) -> None:
        """Setup HTTP session."""
        timeout = aiohttp.ClientTimeout(
            total=self.config.request_timeout,
            connect=self.config.connection_timeout,
        )

        self.session = aiohttp.ClientSession(
            base_url=self.config.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def teardown(self) -> None:
        """End all calls and cleanup."""
        for call_id in self.active_call_ids:
            try:
                async with self.session.post(f"/api/v1/calls/{call_id}/end"):
                    pass
            except Exception:
                pass

        if self.session:
            await self.session.close()

    async def run_iteration(self, user_id: int, iteration: int) -> RequestResult:
        """Manage concurrent calls."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        request_id = f"concurrent-{user_id}-{iteration}"
        start_time = time.time()

        try:
            current_calls = len(self.active_call_ids)

            # Increase target every 10 iterations
            if iteration % 10 == 0:
                self.target_concurrent = min(
                    self.target_concurrent + 5,
                    self.config.concurrent_users,
                )

            if current_calls < self.target_concurrent:
                # Start new call
                payload = {
                    "agent_id": f"agent_{uuid4().hex[:12]}",
                    "phone_number": f"+1555{random.randint(1000000, 9999999)}",
                    "direction": "outbound",
                }

                async with self.session.post("/api/v1/calls", json=payload) as response:
                    response_time = (time.time() - start_time) * 1000

                    if response.status == 200:
                        body = await response.json()
                        call_id = body.get("id", "")
                        self.active_call_ids.append(call_id)
                        self.max_concurrent_calls = max(
                            self.max_concurrent_calls,
                            len(self.active_call_ids),
                        )

                    return RequestResult(
                        request_id=request_id,
                        endpoint="/api/v1/calls",
                        method="POST",
                        status_code=response.status,
                        response_time_ms=response_time,
                        success=response.status == 200,
                        metadata={"concurrent_calls": len(self.active_call_ids)},
                    )

            elif current_calls > self.target_concurrent and self.active_call_ids:
                # End excess calls
                call_id = self.active_call_ids.pop(0)

                async with self.session.post(f"/api/v1/calls/{call_id}/end") as response:
                    response_time = (time.time() - start_time) * 1000

                    return RequestResult(
                        request_id=request_id,
                        endpoint=f"/api/v1/calls/{call_id}/end",
                        method="POST",
                        status_code=response.status,
                        response_time_ms=response_time,
                        success=response.status in (200, 202, 204),
                        metadata={"concurrent_calls": len(self.active_call_ids)},
                    )

            else:
                # Send keep-alive for random call
                if self.active_call_ids:
                    call_id = random.choice(self.active_call_ids)

                    async with self.session.get(f"/api/v1/calls/{call_id}") as response:
                        response_time = (time.time() - start_time) * 1000

                        return RequestResult(
                            request_id=request_id,
                            endpoint=f"/api/v1/calls/{call_id}",
                            method="GET",
                            status_code=response.status,
                            response_time_ms=response_time,
                            success=response.status == 200,
                            metadata={"concurrent_calls": len(self.active_call_ids)},
                        )

                return RequestResult(
                    request_id=request_id,
                    endpoint="noop",
                    method="GET",
                    status_code=200,
                    response_time_ms=0,
                    success=True,
                )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint="concurrent_calls",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )


class DatabaseLoadTest(LoadTestBase):
    """
    Load test for database operations.

    Tests:
    - Read performance
    - Write performance
    - Query complexity handling
    - Connection pool behavior
    """

    def __init__(self, config: LoadTestConfig):
        super().__init__(config)
        self.pool = None

    async def setup(self) -> None:
        """Setup database connection pool."""
        try:
            import asyncpg

            self.pool = await asyncpg.create_pool(
                self.config.database_url or "postgresql://localhost/builderengine",
                min_size=5,
                max_size=self.config.concurrent_users,
            )
        except ImportError:
            logger.warning("asyncpg not installed, using mock database tests")
        except Exception as e:
            logger.warning(f"Could not connect to database: {e}")

    async def teardown(self) -> None:
        """Close database pool."""
        if self.pool:
            await self.pool.close()

    async def run_iteration(self, user_id: int, iteration: int) -> RequestResult:
        """Run a database operation."""
        request_id = f"db-{user_id}-{iteration}"
        start_time = time.time()

        if not self.pool:
            # Return mock result if no database
            await asyncio.sleep(random.uniform(0.001, 0.01))
            return RequestResult(
                request_id=request_id,
                endpoint="database/mock",
                method="QUERY",
                status_code=200,
                response_time_ms=(time.time() - start_time) * 1000,
                success=True,
            )

        try:
            # Select operation based on weights
            operation = self._select_operation()

            if operation == "read_simple":
                return await self._read_simple(request_id)
            elif operation == "read_complex":
                return await self._read_complex(request_id)
            elif operation == "write":
                return await self._write(request_id)
            elif operation == "aggregate":
                return await self._aggregate(request_id)
            else:
                return await self._read_simple(request_id)

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint="database/error",
                method="QUERY",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    def _select_operation(self) -> str:
        """Select operation type."""
        r = random.random()
        if r < 0.50:
            return "read_simple"
        elif r < 0.75:
            return "read_complex"
        elif r < 0.90:
            return "write"
        else:
            return "aggregate"

    async def _read_simple(self, request_id: str) -> RequestResult:
        """Simple read query."""
        start_time = time.time()

        async with self.pool.acquire() as conn:
            await conn.fetchrow(
                "SELECT id, name, created_at FROM agents LIMIT 1"
            )

        return RequestResult(
            request_id=request_id,
            endpoint="database/read_simple",
            method="SELECT",
            status_code=200,
            response_time_ms=(time.time() - start_time) * 1000,
            success=True,
        )

    async def _read_complex(self, request_id: str) -> RequestResult:
        """Complex read with joins."""
        start_time = time.time()

        async with self.pool.acquire() as conn:
            await conn.fetch("""
                SELECT a.id, a.name, COUNT(c.id) as call_count
                FROM agents a
                LEFT JOIN calls c ON c.agent_id = a.id
                WHERE a.organization_id = $1
                GROUP BY a.id, a.name
                ORDER BY call_count DESC
                LIMIT 10
            """, self.config.organization_id or "test")

        return RequestResult(
            request_id=request_id,
            endpoint="database/read_complex",
            method="SELECT",
            status_code=200,
            response_time_ms=(time.time() - start_time) * 1000,
            success=True,
        )

    async def _write(self, request_id: str) -> RequestResult:
        """Write operation."""
        start_time = time.time()

        async with self.pool.acquire() as conn:
            # Use a test table or transaction rollback
            async with conn.transaction():
                await conn.execute("""
                    INSERT INTO test_load (data, created_at)
                    VALUES ($1, NOW())
                """, json.dumps({"test": True}))

                # Rollback to not pollute data
                raise Exception("Rollback")

        return RequestResult(
            request_id=request_id,
            endpoint="database/write",
            method="INSERT",
            status_code=200,
            response_time_ms=(time.time() - start_time) * 1000,
            success=True,
        )

    async def _aggregate(self, request_id: str) -> RequestResult:
        """Aggregate query."""
        start_time = time.time()

        async with self.pool.acquire() as conn:
            await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_calls,
                    AVG(duration_seconds) as avg_duration,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
                FROM calls
                WHERE created_at > NOW() - INTERVAL '1 day'
            """)

        return RequestResult(
            request_id=request_id,
            endpoint="database/aggregate",
            method="SELECT",
            status_code=200,
            response_time_ms=(time.time() - start_time) * 1000,
            success=True,
        )


class MixedWorkloadTest(LoadTestBase):
    """
    Mixed workload test simulating real-world usage.

    Combines:
    - API requests
    - WebSocket connections
    - Call simulation
    """

    def __init__(self, config: LoadTestConfig):
        super().__init__(config)
        self.api_test = APILoadTest(config)
        self.ws_test = WebSocketLoadTest(config)
        self.call_test = CallSimulationTest(config)

    async def setup(self) -> None:
        """Setup all sub-tests."""
        await self.api_test.setup()
        await self.ws_test.setup()
        await self.call_test.setup()

    async def teardown(self) -> None:
        """Teardown all sub-tests."""
        await self.api_test.teardown()
        await self.ws_test.teardown()
        await self.call_test.teardown()

    async def run_iteration(self, user_id: int, iteration: int) -> RequestResult:
        """Run mixed workload iteration."""
        # Distribute users across test types
        test_type = user_id % 3

        if test_type == 0:
            return await self.api_test.run_iteration(user_id, iteration)
        elif test_type == 1:
            return await self.ws_test.run_iteration(user_id, iteration)
        else:
            return await self.call_test.run_iteration(user_id, iteration)

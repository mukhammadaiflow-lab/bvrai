"""Load test scenarios for telephony operations."""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4
import logging

import aiohttp

from .base import LoadTestBase, LoadTestConfig, RequestResult

logger = logging.getLogger(__name__)


class TelephonyLoadTest(LoadTestBase):
    """
    Load test for telephony operations.

    Tests:
    - Outbound call initiation
    - Call control operations (hold, transfer)
    - Recording operations
    - Concurrent call capacity
    """

    def __init__(self, config: LoadTestConfig):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_calls: Dict[int, Dict[str, Any]] = {}
        self.call_stats: Dict[str, int] = {
            "initiated": 0,
            "answered": 0,
            "completed": 0,
            "failed": 0,
            "transferred": 0,
        }

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
        # End any remaining calls
        for user_id, call_data in list(self.active_calls.items()):
            try:
                call_id = call_data.get("call_id")
                if call_id and self.session:
                    async with self.session.post(f"/api/v1/calls/{call_id}/end"):
                        pass
            except Exception:
                pass

        self.active_calls.clear()

        if self.session:
            await self.session.close()

        # Log final stats
        logger.info(f"Telephony load test stats: {self.call_stats}")

    async def run_iteration(self, user_id: int, iteration: int) -> RequestResult:
        """Run a telephony test iteration."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        request_id = f"tel-{user_id}-{iteration}"

        # Select operation based on weighted distribution
        operation = self._select_operation(user_id)

        if operation == "initiate":
            return await self._initiate_call(user_id, request_id)
        elif operation == "status":
            return await self._get_call_status(user_id, request_id)
        elif operation == "hold":
            return await self._hold_call(user_id, request_id)
        elif operation == "resume":
            return await self._resume_call(user_id, request_id)
        elif operation == "transfer":
            return await self._transfer_call(user_id, request_id)
        elif operation == "recording":
            return await self._start_recording(user_id, request_id)
        elif operation == "end":
            return await self._end_call(user_id, request_id)
        else:
            return await self._initiate_call(user_id, request_id)

    def _select_operation(self, user_id: int) -> str:
        """Select operation based on current state and weights."""
        has_active_call = user_id in self.active_calls

        if not has_active_call:
            return "initiate"

        # Weighted selection for active calls
        r = random.random()
        if r < 0.30:
            return "status"
        elif r < 0.45:
            return "hold"
        elif r < 0.55:
            return "resume"
        elif r < 0.65:
            return "transfer"
        elif r < 0.75:
            return "recording"
        else:
            return "end"

    async def _initiate_call(self, user_id: int, request_id: str) -> RequestResult:
        """Initiate a new outbound call."""
        start_time = time.time()

        payload = {
            "agent_id": str(uuid4()),
            "to_number": f"+1555{random.randint(1000000, 9999999)}",
            "from_number": "+15555550000",
            "metadata": {"user_id": user_id, "test": True, "load_test": True},
        }

        try:
            async with self.session.post("/api/v1/calls/outbound", json=payload) as response:
                response_time = (time.time() - start_time) * 1000
                body = await response.json() if response.status in (200, 201) else {}

                if response.status in (200, 201):
                    call_id = body.get("call_id", "")
                    session_id = body.get("session_id", "")
                    self.active_calls[user_id] = {
                        "call_id": str(call_id),
                        "session_id": session_id,
                        "started_at": time.time(),
                        "on_hold": False,
                    }
                    self.call_stats["initiated"] += 1

                return RequestResult(
                    request_id=request_id,
                    endpoint="/api/v1/calls/outbound",
                    method="POST",
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=response.status in (200, 201),
                    response_size_bytes=len(json.dumps(body)),
                    metadata={"operation": "initiate"},
                )

        except Exception as e:
            self.call_stats["failed"] += 1
            return RequestResult(
                request_id=request_id,
                endpoint="/api/v1/calls/outbound",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def _get_call_status(self, user_id: int, request_id: str) -> RequestResult:
        """Get status of active call."""
        start_time = time.time()

        call_data = self.active_calls.get(user_id)
        if not call_data:
            return await self._initiate_call(user_id, request_id)

        call_id = call_data["call_id"]

        try:
            async with self.session.get(f"/api/v1/calls/{call_id}/status") as response:
                response_time = (time.time() - start_time) * 1000
                body = await response.read()

                return RequestResult(
                    request_id=request_id,
                    endpoint=f"/api/v1/calls/{call_id}/status",
                    method="GET",
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=response.status == 200,
                    response_size_bytes=len(body),
                    metadata={"operation": "status"},
                )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint=f"/api/v1/calls/{call_id}/status",
                method="GET",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def _hold_call(self, user_id: int, request_id: str) -> RequestResult:
        """Put call on hold."""
        start_time = time.time()

        call_data = self.active_calls.get(user_id)
        if not call_data or call_data.get("on_hold"):
            return await self._get_call_status(user_id, request_id)

        call_id = call_data["call_id"]

        try:
            async with self.session.post(f"/api/v1/calls/{call_id}/hold") as response:
                response_time = (time.time() - start_time) * 1000

                if response.status == 200:
                    self.active_calls[user_id]["on_hold"] = True

                return RequestResult(
                    request_id=request_id,
                    endpoint=f"/api/v1/calls/{call_id}/hold",
                    method="POST",
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=response.status == 200,
                    metadata={"operation": "hold"},
                )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint=f"/api/v1/calls/{call_id}/hold",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def _resume_call(self, user_id: int, request_id: str) -> RequestResult:
        """Resume call from hold."""
        start_time = time.time()

        call_data = self.active_calls.get(user_id)
        if not call_data or not call_data.get("on_hold"):
            return await self._get_call_status(user_id, request_id)

        call_id = call_data["call_id"]

        try:
            async with self.session.post(f"/api/v1/calls/{call_id}/resume") as response:
                response_time = (time.time() - start_time) * 1000

                if response.status == 200:
                    self.active_calls[user_id]["on_hold"] = False

                return RequestResult(
                    request_id=request_id,
                    endpoint=f"/api/v1/calls/{call_id}/resume",
                    method="POST",
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=response.status == 200,
                    metadata={"operation": "resume"},
                )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint=f"/api/v1/calls/{call_id}/resume",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def _transfer_call(self, user_id: int, request_id: str) -> RequestResult:
        """Transfer call to another number."""
        start_time = time.time()

        call_data = self.active_calls.get(user_id)
        if not call_data:
            return await self._initiate_call(user_id, request_id)

        call_id = call_data["call_id"]

        payload = {
            "target_number": f"+1555{random.randint(1000000, 9999999)}",
            "announce": True,
            "message": "Transferring your call.",
        }

        try:
            async with self.session.post(
                f"/api/v1/calls/{call_id}/transfer",
                json=payload,
            ) as response:
                response_time = (time.time() - start_time) * 1000

                if response.status == 200:
                    self.call_stats["transferred"] += 1
                    # Remove from active calls as it's transferred
                    self.active_calls.pop(user_id, None)

                return RequestResult(
                    request_id=request_id,
                    endpoint=f"/api/v1/calls/{call_id}/transfer",
                    method="POST",
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=response.status == 200,
                    metadata={"operation": "transfer"},
                )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint=f"/api/v1/calls/{call_id}/transfer",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def _start_recording(self, user_id: int, request_id: str) -> RequestResult:
        """Start recording active call."""
        start_time = time.time()

        call_data = self.active_calls.get(user_id)
        if not call_data:
            return await self._get_call_status(user_id, request_id)

        call_id = call_data["call_id"]

        try:
            async with self.session.post(f"/api/v1/calls/{call_id}/recording/start") as response:
                response_time = (time.time() - start_time) * 1000

                return RequestResult(
                    request_id=request_id,
                    endpoint=f"/api/v1/calls/{call_id}/recording/start",
                    method="POST",
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=response.status == 200,
                    metadata={"operation": "recording"},
                )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint=f"/api/v1/calls/{call_id}/recording/start",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def _end_call(self, user_id: int, request_id: str) -> RequestResult:
        """End active call."""
        start_time = time.time()

        call_data = self.active_calls.pop(user_id, None)
        if not call_data:
            return await self._initiate_call(user_id, request_id)

        call_id = call_data["call_id"]

        try:
            async with self.session.post(f"/api/v1/calls/{call_id}/end") as response:
                response_time = (time.time() - start_time) * 1000

                if response.status in (200, 202, 204):
                    self.call_stats["completed"] += 1

                return RequestResult(
                    request_id=request_id,
                    endpoint=f"/api/v1/calls/{call_id}/end",
                    method="POST",
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=response.status in (200, 202, 204),
                    metadata={"operation": "end"},
                )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint=f"/api/v1/calls/{call_id}/end",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )


class CallTransferLoadTest(LoadTestBase):
    """
    Load test specifically for call transfer operations.

    Tests:
    - Blind transfers
    - Attended transfers
    - Conference transfers
    - Transfer failure handling
    """

    def __init__(self, config: LoadTestConfig):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_calls: Dict[int, str] = {}

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
        """Cleanup."""
        if self.session:
            await self.session.close()

    async def run_iteration(self, user_id: int, iteration: int) -> RequestResult:
        """Run transfer test iteration."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        request_id = f"transfer-{user_id}-{iteration}"
        start_time = time.time()

        # First initiate a call if we don't have one
        if user_id not in self.active_calls:
            result = await self._initiate_call_for_transfer(user_id, request_id)
            if not result.success:
                return result

        # Now perform transfer
        return await self._perform_transfer(user_id, request_id)

    async def _initiate_call_for_transfer(
        self,
        user_id: int,
        request_id: str,
    ) -> RequestResult:
        """Initiate call for transfer testing."""
        start_time = time.time()

        payload = {
            "agent_id": str(uuid4()),
            "to_number": f"+1555{random.randint(1000000, 9999999)}",
            "from_number": "+15555550000",
            "metadata": {"transfer_test": True},
        }

        try:
            async with self.session.post("/api/v1/calls/outbound", json=payload) as response:
                response_time = (time.time() - start_time) * 1000
                body = await response.json() if response.status in (200, 201) else {}

                if response.status in (200, 201):
                    self.active_calls[user_id] = str(body.get("call_id", ""))

                return RequestResult(
                    request_id=request_id,
                    endpoint="/api/v1/calls/outbound",
                    method="POST",
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=response.status in (200, 201),
                )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint="/api/v1/calls/outbound",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def _perform_transfer(self, user_id: int, request_id: str) -> RequestResult:
        """Perform call transfer."""
        start_time = time.time()

        call_id = self.active_calls.get(user_id)
        if not call_id:
            return await self._initiate_call_for_transfer(user_id, request_id)

        payload = {
            "target_number": f"+1555{random.randint(1000000, 9999999)}",
            "announce": random.choice([True, False]),
            "message": "Transferring..." if random.random() > 0.5 else None,
        }

        try:
            async with self.session.post(
                f"/api/v1/calls/{call_id}/transfer",
                json=payload,
            ) as response:
                response_time = (time.time() - start_time) * 1000

                # Remove from active calls after transfer
                if response.status == 200:
                    self.active_calls.pop(user_id, None)

                return RequestResult(
                    request_id=request_id,
                    endpoint=f"/api/v1/calls/{call_id}/transfer",
                    method="POST",
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=response.status == 200,
                )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint=f"/api/v1/calls/{call_id}/transfer",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )


class RecordingLoadTest(LoadTestBase):
    """
    Load test for call recording operations.

    Tests:
    - Start recording during call
    - Stop recording
    - Concurrent recording operations
    """

    def __init__(self, config: LoadTestConfig):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.calls_with_recordings: Dict[int, Dict[str, str]] = {}

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
        """Cleanup."""
        if self.session:
            await self.session.close()

    async def run_iteration(self, user_id: int, iteration: int) -> RequestResult:
        """Run recording test iteration."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        request_id = f"rec-{user_id}-{iteration}"

        # Cycle through operations
        if iteration % 3 == 0:
            return await self._start_recording(user_id, request_id)
        elif iteration % 3 == 1:
            return await self._stop_recording(user_id, request_id)
        else:
            return await self._get_recording(user_id, request_id)

    async def _start_recording(self, user_id: int, request_id: str) -> RequestResult:
        """Start recording on a call."""
        start_time = time.time()

        # First need a call
        if user_id not in self.calls_with_recordings:
            # Create call
            payload = {
                "agent_id": str(uuid4()),
                "to_number": f"+1555{random.randint(1000000, 9999999)}",
                "from_number": "+15555550000",
            }

            async with self.session.post("/api/v1/calls/outbound", json=payload) as response:
                if response.status in (200, 201):
                    body = await response.json()
                    self.calls_with_recordings[user_id] = {
                        "call_id": str(body.get("call_id", "")),
                        "recording_sid": "",
                    }

        call_data = self.calls_with_recordings.get(user_id, {})
        call_id = call_data.get("call_id")

        if not call_id:
            return RequestResult(
                request_id=request_id,
                endpoint="/api/v1/calls/*/recording/start",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message="No call available",
            )

        try:
            async with self.session.post(f"/api/v1/calls/{call_id}/recording/start") as response:
                response_time = (time.time() - start_time) * 1000
                body = await response.json() if response.status == 200 else {}

                if response.status == 200:
                    self.calls_with_recordings[user_id]["recording_sid"] = body.get(
                        "recording_sid", ""
                    )

                return RequestResult(
                    request_id=request_id,
                    endpoint=f"/api/v1/calls/{call_id}/recording/start",
                    method="POST",
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=response.status == 200,
                )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint=f"/api/v1/calls/{call_id}/recording/start",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def _stop_recording(self, user_id: int, request_id: str) -> RequestResult:
        """Stop recording on a call."""
        start_time = time.time()

        call_data = self.calls_with_recordings.get(user_id, {})
        call_id = call_data.get("call_id")
        recording_sid = call_data.get("recording_sid")

        if not call_id or not recording_sid:
            return await self._start_recording(user_id, request_id)

        try:
            async with self.session.post(
                f"/api/v1/calls/{call_id}/recording/stop",
                params={"recording_sid": recording_sid},
            ) as response:
                response_time = (time.time() - start_time) * 1000

                if response.status == 200:
                    self.calls_with_recordings[user_id]["recording_sid"] = ""

                return RequestResult(
                    request_id=request_id,
                    endpoint=f"/api/v1/calls/{call_id}/recording/stop",
                    method="POST",
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=response.status == 200,
                )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint=f"/api/v1/calls/{call_id}/recording/stop",
                method="POST",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def _get_recording(self, user_id: int, request_id: str) -> RequestResult:
        """Get recording for a call."""
        start_time = time.time()

        call_data = self.calls_with_recordings.get(user_id, {})
        call_id = call_data.get("call_id")

        if not call_id:
            return await self._start_recording(user_id, request_id)

        try:
            async with self.session.get(f"/api/v1/calls/{call_id}/recording") as response:
                response_time = (time.time() - start_time) * 1000
                body = await response.read()

                return RequestResult(
                    request_id=request_id,
                    endpoint=f"/api/v1/calls/{call_id}/recording",
                    method="GET",
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=response.status == 200,
                    response_size_bytes=len(body),
                )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                endpoint=f"/api/v1/calls/{call_id}/recording",
                method="GET",
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

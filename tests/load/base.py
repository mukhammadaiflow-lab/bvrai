"""Base classes for load testing."""

import asyncio
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LoadTestStatus(str, Enum):
    """Load test status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class LoadTestConfig:
    """Configuration for load tests."""
    # Target
    base_url: str = "http://localhost:8000"
    api_key: str = ""
    organization_id: str = ""

    # Load profile
    concurrent_users: int = 10
    requests_per_second: int = 100
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    ramp_down_seconds: int = 5

    # Timeouts
    request_timeout: float = 30.0
    connection_timeout: float = 10.0

    # Thresholds (for pass/fail)
    max_avg_response_time_ms: float = 500.0
    max_p95_response_time_ms: float = 1000.0
    max_p99_response_time_ms: float = 2000.0
    max_error_rate: float = 0.01  # 1%
    min_requests_per_second: float = 50.0

    # WebSocket specific
    websocket_url: str = ""
    websocket_reconnect_attempts: int = 3

    # Database specific
    database_url: str = ""

    # Output
    output_dir: str = "./load_test_results"
    save_detailed_results: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "organization_id": self.organization_id,
            "concurrent_users": self.concurrent_users,
            "requests_per_second": self.requests_per_second,
            "duration_seconds": self.duration_seconds,
            "ramp_up_seconds": self.ramp_up_seconds,
            "ramp_down_seconds": self.ramp_down_seconds,
            "request_timeout": self.request_timeout,
            "max_avg_response_time_ms": self.max_avg_response_time_ms,
            "max_p95_response_time_ms": self.max_p95_response_time_ms,
            "max_p99_response_time_ms": self.max_p99_response_time_ms,
            "max_error_rate": self.max_error_rate,
            "min_requests_per_second": self.min_requests_per_second,
        }


@dataclass
class RequestResult:
    """Result of a single request."""
    request_id: str
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    success: bool
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    response_size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadTestResult:
    """Complete load test results."""
    test_name: str
    config: LoadTestConfig
    status: LoadTestStatus = LoadTestStatus.PENDING

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Request statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Response time statistics (milliseconds)
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    median_response_time_ms: float = 0.0
    p90_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    std_dev_response_time_ms: float = 0.0

    # Throughput
    requests_per_second: float = 0.0
    bytes_per_second: float = 0.0

    # Error analysis
    error_rate: float = 0.0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    errors_by_endpoint: Dict[str, int] = field(default_factory=dict)

    # Response codes
    response_codes: Dict[int, int] = field(default_factory=dict)

    # Per-endpoint statistics
    endpoint_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Detailed results (if enabled)
    detailed_results: List[RequestResult] = field(default_factory=list)

    # Time series data for graphs
    time_series: List[Dict[str, Any]] = field(default_factory=list)

    # Pass/fail
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "min_response_time_ms": self.min_response_time_ms,
            "max_response_time_ms": self.max_response_time_ms,
            "avg_response_time_ms": self.avg_response_time_ms,
            "median_response_time_ms": self.median_response_time_ms,
            "p90_response_time_ms": self.p90_response_time_ms,
            "p95_response_time_ms": self.p95_response_time_ms,
            "p99_response_time_ms": self.p99_response_time_ms,
            "std_dev_response_time_ms": self.std_dev_response_time_ms,
            "requests_per_second": self.requests_per_second,
            "error_rate": self.error_rate,
            "errors_by_type": self.errors_by_type,
            "response_codes": self.response_codes,
            "endpoint_stats": self.endpoint_stats,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
        }

    def calculate_statistics(self, response_times: List[float]) -> None:
        """Calculate statistics from response times."""
        if not response_times:
            return

        sorted_times = sorted(response_times)
        n = len(sorted_times)

        self.min_response_time_ms = sorted_times[0]
        self.max_response_time_ms = sorted_times[-1]
        self.avg_response_time_ms = statistics.mean(sorted_times)
        self.median_response_time_ms = statistics.median(sorted_times)
        self.std_dev_response_time_ms = statistics.stdev(sorted_times) if n > 1 else 0.0

        # Percentiles
        self.p90_response_time_ms = sorted_times[int(n * 0.90)]
        self.p95_response_time_ms = sorted_times[int(n * 0.95)]
        self.p99_response_time_ms = sorted_times[int(n * 0.99)] if n >= 100 else sorted_times[-1]

    def evaluate_thresholds(self, config: LoadTestConfig) -> None:
        """Evaluate results against configured thresholds."""
        self.failure_reasons = []

        if self.avg_response_time_ms > config.max_avg_response_time_ms:
            self.failure_reasons.append(
                f"Average response time ({self.avg_response_time_ms:.2f}ms) exceeds threshold ({config.max_avg_response_time_ms}ms)"
            )

        if self.p95_response_time_ms > config.max_p95_response_time_ms:
            self.failure_reasons.append(
                f"P95 response time ({self.p95_response_time_ms:.2f}ms) exceeds threshold ({config.max_p95_response_time_ms}ms)"
            )

        if self.p99_response_time_ms > config.max_p99_response_time_ms:
            self.failure_reasons.append(
                f"P99 response time ({self.p99_response_time_ms:.2f}ms) exceeds threshold ({config.max_p99_response_time_ms}ms)"
            )

        if self.error_rate > config.max_error_rate:
            self.failure_reasons.append(
                f"Error rate ({self.error_rate:.2%}) exceeds threshold ({config.max_error_rate:.2%})"
            )

        if self.requests_per_second < config.min_requests_per_second:
            self.failure_reasons.append(
                f"Throughput ({self.requests_per_second:.2f} req/s) below threshold ({config.min_requests_per_second} req/s)"
            )

        self.passed = len(self.failure_reasons) == 0


class LoadTestBase(ABC):
    """Base class for load tests."""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.result = LoadTestResult(
            test_name=self.__class__.__name__,
            config=config,
        )
        self._stop_requested = False
        self._running = False

    @abstractmethod
    async def setup(self) -> None:
        """Setup before running the test."""
        pass

    @abstractmethod
    async def teardown(self) -> None:
        """Cleanup after the test."""
        pass

    @abstractmethod
    async def run_iteration(self, user_id: int, iteration: int) -> RequestResult:
        """Run a single test iteration."""
        pass

    async def run(self) -> LoadTestResult:
        """Run the load test."""
        logger.info(f"Starting load test: {self.__class__.__name__}")

        try:
            self._running = True
            self.result.status = LoadTestStatus.RUNNING
            self.result.started_at = datetime.utcnow()

            # Setup
            await self.setup()

            # Run test
            await self._execute_load_test()

            # Calculate final statistics
            self._finalize_results()

            self.result.status = LoadTestStatus.COMPLETED

        except Exception as e:
            logger.error(f"Load test failed: {e}")
            self.result.status = LoadTestStatus.FAILED
            self.result.failure_reasons.append(str(e))

        finally:
            self._running = False
            await self.teardown()
            self.result.completed_at = datetime.utcnow()

            if self.result.started_at:
                self.result.duration_seconds = (
                    self.result.completed_at - self.result.started_at
                ).total_seconds()

        logger.info(f"Load test completed: {self.result.status.value}")
        return self.result

    async def _execute_load_test(self) -> None:
        """Execute the main load test loop."""
        response_times: List[float] = []
        total_bytes = 0
        time_series_data: List[Dict[str, Any]] = []

        start_time = time.time()
        end_time = start_time + self.config.duration_seconds

        # Create tasks for concurrent users
        tasks = []
        iteration_counts = [0] * self.config.concurrent_users

        # Time series interval
        interval_start = start_time
        interval_results: List[RequestResult] = []

        while time.time() < end_time and not self._stop_requested:
            elapsed = time.time() - start_time
            current_users = self._get_current_users(elapsed)

            # Launch requests for current users
            for user_id in range(current_users):
                iteration = iteration_counts[user_id]
                iteration_counts[user_id] += 1

                task = asyncio.create_task(
                    self._run_iteration_safe(user_id, iteration)
                )
                tasks.append(task)

            # Process completed tasks
            done, pending = await asyncio.wait(
                tasks,
                timeout=0.1,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                tasks.remove(task)
                try:
                    result = task.result()
                    if result:
                        self._process_result(result, response_times)
                        total_bytes += result.response_size_bytes
                        interval_results.append(result)

                        if self.config.save_detailed_results:
                            self.result.detailed_results.append(result)
                except Exception as e:
                    logger.debug(f"Task error: {e}")

            # Record time series data every second
            if time.time() - interval_start >= 1.0:
                time_series_data.append(self._create_time_series_entry(
                    elapsed,
                    interval_results,
                ))
                interval_results = []
                interval_start = time.time()

            # Rate limiting
            await self._apply_rate_limit(elapsed)

        # Wait for remaining tasks
        if tasks:
            done, _ = await asyncio.wait(tasks, timeout=self.config.request_timeout)
            for task in done:
                try:
                    result = task.result()
                    if result:
                        self._process_result(result, response_times)
                except Exception:
                    pass

        # Store time series
        self.result.time_series = time_series_data

        # Calculate statistics
        self.result.calculate_statistics(response_times)

        # Calculate throughput
        total_time = time.time() - start_time
        if total_time > 0:
            self.result.requests_per_second = self.result.total_requests / total_time
            self.result.bytes_per_second = total_bytes / total_time

    def _get_current_users(self, elapsed: float) -> int:
        """Get number of concurrent users based on ramp-up/down."""
        if elapsed < self.config.ramp_up_seconds:
            # Ramp up
            progress = elapsed / self.config.ramp_up_seconds
            return max(1, int(self.config.concurrent_users * progress))

        remaining = self.config.duration_seconds - elapsed
        if remaining < self.config.ramp_down_seconds:
            # Ramp down
            progress = remaining / self.config.ramp_down_seconds
            return max(1, int(self.config.concurrent_users * progress))

        return self.config.concurrent_users

    async def _run_iteration_safe(
        self,
        user_id: int,
        iteration: int,
    ) -> Optional[RequestResult]:
        """Run iteration with error handling."""
        try:
            return await asyncio.wait_for(
                self.run_iteration(user_id, iteration),
                timeout=self.config.request_timeout,
            )
        except asyncio.TimeoutError:
            return RequestResult(
                request_id=f"{user_id}-{iteration}",
                endpoint="timeout",
                method="",
                status_code=0,
                response_time_ms=self.config.request_timeout * 1000,
                success=False,
                error_message="Request timeout",
            )
        except Exception as e:
            return RequestResult(
                request_id=f"{user_id}-{iteration}",
                endpoint="error",
                method="",
                status_code=0,
                response_time_ms=0,
                success=False,
                error_message=str(e),
            )

    def _process_result(
        self,
        result: RequestResult,
        response_times: List[float],
    ) -> None:
        """Process a single request result."""
        self.result.total_requests += 1

        if result.success:
            self.result.successful_requests += 1
            response_times.append(result.response_time_ms)
        else:
            self.result.failed_requests += 1

            # Track errors by type
            error_type = result.error_message.split(":")[0] if result.error_message else "Unknown"
            self.result.errors_by_type[error_type] = (
                self.result.errors_by_type.get(error_type, 0) + 1
            )

            # Track errors by endpoint
            self.result.errors_by_endpoint[result.endpoint] = (
                self.result.errors_by_endpoint.get(result.endpoint, 0) + 1
            )

        # Track response codes
        if result.status_code:
            self.result.response_codes[result.status_code] = (
                self.result.response_codes.get(result.status_code, 0) + 1
            )

        # Track per-endpoint stats
        if result.endpoint not in self.result.endpoint_stats:
            self.result.endpoint_stats[result.endpoint] = {
                "count": 0,
                "success": 0,
                "failed": 0,
                "response_times": [],
            }

        stats = self.result.endpoint_stats[result.endpoint]
        stats["count"] += 1
        if result.success:
            stats["success"] += 1
            stats["response_times"].append(result.response_time_ms)
        else:
            stats["failed"] += 1

    def _create_time_series_entry(
        self,
        elapsed: float,
        results: List[RequestResult],
    ) -> Dict[str, Any]:
        """Create a time series entry."""
        success = sum(1 for r in results if r.success)
        failed = len(results) - success
        response_times = [r.response_time_ms for r in results if r.success]

        return {
            "elapsed_seconds": elapsed,
            "requests": len(results),
            "success": success,
            "failed": failed,
            "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "p95_response_time_ms": sorted(response_times)[int(len(response_times) * 0.95)] if len(response_times) > 20 else max(response_times) if response_times else 0,
            "requests_per_second": len(results),
            "error_rate": failed / len(results) if results else 0,
        }

    async def _apply_rate_limit(self, elapsed: float) -> None:
        """Apply rate limiting based on requests per second target."""
        expected_requests = elapsed * self.config.requests_per_second
        actual_requests = self.result.total_requests

        if actual_requests > expected_requests:
            delay = (actual_requests - expected_requests) / self.config.requests_per_second
            await asyncio.sleep(min(delay, 0.1))
        else:
            await asyncio.sleep(0.01)  # Small delay to prevent tight loop

    def _finalize_results(self) -> None:
        """Finalize results and calculate derived metrics."""
        if self.result.total_requests > 0:
            self.result.error_rate = self.result.failed_requests / self.result.total_requests

        # Calculate per-endpoint statistics
        for endpoint, stats in self.result.endpoint_stats.items():
            times = stats.pop("response_times", [])
            if times:
                sorted_times = sorted(times)
                stats["avg_response_time_ms"] = statistics.mean(times)
                stats["p95_response_time_ms"] = sorted_times[int(len(times) * 0.95)]
                stats["error_rate"] = stats["failed"] / stats["count"] if stats["count"] > 0 else 0

        # Evaluate against thresholds
        self.result.evaluate_thresholds(self.config)

    def stop(self) -> None:
        """Request test to stop."""
        self._stop_requested = True

    @property
    def is_running(self) -> bool:
        return self._running


class VirtualUser:
    """Represents a virtual user in load testing."""

    def __init__(self, user_id: int, config: LoadTestConfig):
        self.user_id = user_id
        self.config = config
        self.session_data: Dict[str, Any] = {}
        self.request_count = 0

    async def think(self, min_seconds: float = 0.1, max_seconds: float = 1.0) -> None:
        """Simulate user think time."""
        import random
        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)

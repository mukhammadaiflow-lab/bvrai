"""Load test runner for executing and managing load tests."""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
from pathlib import Path

from .base import LoadTestBase, LoadTestConfig, LoadTestResult, LoadTestStatus

logger = logging.getLogger(__name__)


class LoadTestRunner:
    """
    Runner for executing load tests.

    Features:
    - Sequential or parallel test execution
    - Progress reporting
    - Result aggregation
    - Output to various formats
    """

    def __init__(
        self,
        config: LoadTestConfig,
        output_dir: str = "./load_test_results",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tests: List[LoadTestBase] = []
        self.results: List[LoadTestResult] = []
        self._running_tests: List[LoadTestBase] = []

    def add_test(self, test_class: Type[LoadTestBase]) -> None:
        """Add a test to the runner."""
        test = test_class(self.config)
        self.tests.append(test)

    def add_tests(self, test_classes: List[Type[LoadTestBase]]) -> None:
        """Add multiple tests."""
        for test_class in test_classes:
            self.add_test(test_class)

    async def run_all(
        self,
        parallel: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> List[LoadTestResult]:
        """
        Run all added tests.

        Args:
            parallel: Run tests in parallel
            progress_callback: Called with progress updates

        Returns:
            List of test results
        """
        self.results = []
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Starting load test run: {run_id}")
        logger.info(f"Tests to run: {len(self.tests)}")
        logger.info(f"Mode: {'parallel' if parallel else 'sequential'}")

        try:
            if parallel:
                # Run all tests concurrently
                tasks = [
                    self._run_test_with_progress(test, progress_callback)
                    for test in self.tests
                ]
                self.results = await asyncio.gather(*tasks)
            else:
                # Run tests sequentially
                for i, test in enumerate(self.tests):
                    if progress_callback:
                        progress_callback({
                            "type": "test_starting",
                            "test_number": i + 1,
                            "total_tests": len(self.tests),
                            "test_name": test.__class__.__name__,
                        })

                    result = await self._run_test_with_progress(test, progress_callback)
                    self.results.append(result)

        finally:
            # Save results
            await self._save_results(run_id)

        # Generate summary
        summary = self._generate_summary()
        logger.info(f"Load test run completed: {summary['passed']}/{summary['total']} passed")

        return self.results

    async def _run_test_with_progress(
        self,
        test: LoadTestBase,
        progress_callback: Optional[callable],
    ) -> LoadTestResult:
        """Run a single test with progress reporting."""
        self._running_tests.append(test)

        try:
            # Setup progress monitoring
            monitor_task = None
            if progress_callback:
                monitor_task = asyncio.create_task(
                    self._monitor_progress(test, progress_callback)
                )

            # Run the test
            result = await test.run()

            # Cancel monitor
            if monitor_task:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

            # Report completion
            if progress_callback:
                progress_callback({
                    "type": "test_completed",
                    "test_name": test.__class__.__name__,
                    "passed": result.passed,
                    "duration": result.duration_seconds,
                    "requests": result.total_requests,
                    "error_rate": result.error_rate,
                })

            return result

        finally:
            self._running_tests.remove(test)

    async def _monitor_progress(
        self,
        test: LoadTestBase,
        callback: callable,
    ) -> None:
        """Monitor test progress and report updates."""
        while test.is_running:
            callback({
                "type": "progress",
                "test_name": test.__class__.__name__,
                "total_requests": test.result.total_requests,
                "successful_requests": test.result.successful_requests,
                "failed_requests": test.result.failed_requests,
                "avg_response_time": test.result.avg_response_time_ms,
            })
            await asyncio.sleep(1)

    async def _save_results(self, run_id: str) -> None:
        """Save results to files."""
        # Create run directory
        run_dir = self.output_dir / run_id
        run_dir.mkdir(exist_ok=True)

        # Save each test result
        for result in self.results:
            result_file = run_dir / f"{result.test_name}.json"
            with open(result_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

        # Save summary
        summary = self._generate_summary()
        summary_file = run_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Save config
        config_file = run_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Results saved to: {run_dir}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all test results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        total_requests = sum(r.total_requests for r in self.results)
        total_errors = sum(r.failed_requests for r in self.results)
        avg_response_times = [r.avg_response_time_ms for r in self.results if r.avg_response_time_ms > 0]
        p95_response_times = [r.p95_response_time_ms for r in self.results if r.p95_response_time_ms > 0]

        return {
            "run_timestamp": datetime.utcnow().isoformat(),
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "overall_error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "avg_response_time_ms": sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0,
            "avg_p95_response_time_ms": sum(p95_response_times) / len(p95_response_times) if p95_response_times else 0,
            "test_results": [
                {
                    "name": r.test_name,
                    "passed": r.passed,
                    "requests": r.total_requests,
                    "error_rate": r.error_rate,
                    "avg_response_time_ms": r.avg_response_time_ms,
                    "p95_response_time_ms": r.p95_response_time_ms,
                    "failure_reasons": r.failure_reasons,
                }
                for r in self.results
            ],
        }

    def stop_all(self) -> None:
        """Stop all running tests."""
        for test in self._running_tests:
            test.stop()


class LoadTestSuite:
    """
    Pre-configured load test suite.

    Provides standard test scenarios for Builder Engine.
    """

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.runner = LoadTestRunner(config)

    def add_api_tests(self) -> None:
        """Add API load tests."""
        from .scenarios import APILoadTest
        self.runner.add_test(APILoadTest)

    def add_call_tests(self) -> None:
        """Add call simulation tests."""
        from .scenarios import CallSimulationTest
        self.runner.add_test(CallSimulationTest)

    def add_websocket_tests(self) -> None:
        """Add WebSocket tests."""
        from .scenarios import WebSocketLoadTest
        self.runner.add_test(WebSocketLoadTest)

    def add_concurrent_calls_test(self) -> None:
        """Add concurrent calls capacity test."""
        from .scenarios import ConcurrentCallsTest
        self.runner.add_test(ConcurrentCallsTest)

    def add_database_tests(self) -> None:
        """Add database load tests."""
        from .scenarios import DatabaseLoadTest
        self.runner.add_test(DatabaseLoadTest)

    def add_all_tests(self) -> None:
        """Add all standard tests."""
        self.add_api_tests()
        self.add_call_tests()
        self.add_websocket_tests()
        self.add_concurrent_calls_test()
        self.add_database_tests()

    async def run(
        self,
        parallel: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> List[LoadTestResult]:
        """Run the test suite."""
        return await self.runner.run_all(
            parallel=parallel,
            progress_callback=progress_callback,
        )


async def run_quick_test(config: LoadTestConfig) -> LoadTestResult:
    """
    Run a quick smoke test.

    Useful for CI/CD pipelines.
    """
    from .scenarios import APILoadTest

    # Override config for quick test
    quick_config = LoadTestConfig(
        base_url=config.base_url,
        api_key=config.api_key,
        organization_id=config.organization_id,
        concurrent_users=5,
        requests_per_second=10,
        duration_seconds=10,
        ramp_up_seconds=2,
        ramp_down_seconds=2,
    )

    test = APILoadTest(quick_config)
    return await test.run()


async def run_stress_test(config: LoadTestConfig) -> LoadTestResult:
    """
    Run a stress test to find breaking points.

    Gradually increases load until failures occur.
    """
    from .scenarios import ConcurrentCallsTest

    # Override config for stress test
    stress_config = LoadTestConfig(
        base_url=config.base_url,
        api_key=config.api_key,
        organization_id=config.organization_id,
        concurrent_users=100,
        requests_per_second=500,
        duration_seconds=300,
        ramp_up_seconds=60,
        ramp_down_seconds=30,
        max_error_rate=0.1,  # Higher tolerance for stress test
    )

    test = ConcurrentCallsTest(stress_config)
    return await test.run()


async def run_soak_test(
    config: LoadTestConfig,
    duration_hours: int = 1,
) -> LoadTestResult:
    """
    Run a soak test for stability.

    Runs at moderate load for extended period.
    """
    from .scenarios import MixedWorkloadTest

    # Override config for soak test
    soak_config = LoadTestConfig(
        base_url=config.base_url,
        api_key=config.api_key,
        organization_id=config.organization_id,
        concurrent_users=20,
        requests_per_second=50,
        duration_seconds=duration_hours * 3600,
        ramp_up_seconds=60,
        ramp_down_seconds=60,
    )

    test = MixedWorkloadTest(soak_config)
    return await test.run()

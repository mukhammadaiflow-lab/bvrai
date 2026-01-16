"""
Agent Testing & Simulation Framework - Conversation Simulator

Provides comprehensive conversation simulation for testing AI voice agents,
including automated testing, scenario execution, and replay capabilities.
"""

import asyncio
import logging
import random
import time
import uuid
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from .base import (
    AssertionType,
    BenchmarkMetric,
    BenchmarkResult,
    ConversationScript,
    LatencyMetrics,
    SimulatedUser,
    SimulatorMode,
    TestAssertion,
    TestCategory,
    TestError,
    TestMessage,
    TestPriority,
    TestReport,
    TestScenario,
    TestStatus,
    TestStep,
    TestSuite,
    TestTimeoutError,
    VoiceSimulationType,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Agent Interface (Protocol)
# =============================================================================


class AgentInterface:
    """
    Interface for interacting with an AI agent.

    This can be subclassed to connect to different agent implementations.
    """

    async def initialize(
        self,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the agent."""
        pass

    async def send_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a message to the agent and get response.

        Returns dict with:
        - response: str - Agent's response text
        - latency_ms: float - Response latency
        - metadata: dict - Additional metadata (intents, entities, etc.)
        """
        raise NotImplementedError

    async def send_audio(
        self,
        audio_data: bytes,
        format: str = "wav",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send audio to the agent and get response.

        Returns dict with:
        - response: str - Agent's response text
        - audio: bytes - Response audio
        - latency_ms: float - Response latency
        - stt_latency_ms: float - STT processing time
        - tts_latency_ms: float - TTS processing time
        - metadata: dict - Additional metadata
        """
        raise NotImplementedError

    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get current conversation history."""
        return []

    async def reset_conversation(self) -> None:
        """Reset the conversation state."""
        pass

    async def close(self) -> None:
        """Close the agent connection."""
        pass


class MockAgentInterface(AgentInterface):
    """
    Mock agent for testing the testing framework itself.

    Provides configurable responses for development and testing.
    """

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "I understand. How can I help you?",
        latency_range: Tuple[int, int] = (50, 200),
        error_rate: float = 0.0,
    ):
        self.responses = responses or {}
        self.default_response = default_response
        self.latency_range = latency_range
        self.error_rate = error_rate
        self.history: List[Dict[str, Any]] = []

    async def send_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Simulate latency
        latency_ms = random.randint(*self.latency_range)
        await asyncio.sleep(latency_ms / 1000)

        # Simulate errors
        if random.random() < self.error_rate:
            raise TestError("Simulated agent error")

        # Find matching response
        response = self.default_response
        message_lower = message.lower()
        for pattern, resp in self.responses.items():
            if pattern.lower() in message_lower:
                response = resp
                break

        # Store in history
        self.history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return {
            "response": response,
            "latency_ms": latency_ms,
            "metadata": {
                "intent": self._detect_mock_intent(message),
                "entities": [],
                "tool_calls": [],
            },
        }

    def _detect_mock_intent(self, message: str) -> Dict[str, Any]:
        """Simple mock intent detection."""
        message_lower = message.lower()
        if any(w in message_lower for w in ["hi", "hello", "hey"]):
            return {"intent": "greeting", "confidence": 0.95}
        elif any(w in message_lower for w in ["bye", "goodbye", "thanks"]):
            return {"intent": "farewell", "confidence": 0.92}
        elif "?" in message:
            return {"intent": "question", "confidence": 0.85}
        return {"intent": "statement", "confidence": 0.70}

    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        return self.history.copy()

    async def reset_conversation(self) -> None:
        self.history.clear()


# =============================================================================
# Conversation Simulator
# =============================================================================


class ConversationSimulator:
    """
    Simulates conversations with an AI agent for testing purposes.

    Supports multiple modes:
    - Interactive: Step-by-step with optional user input
    - Automated: Fully automated from test scenarios
    - Replay: Replay recorded conversations
    - Random: Generate random but valid inputs
    - Adversarial: Edge cases and stress testing
    """

    def __init__(
        self,
        agent: AgentInterface,
        mode: SimulatorMode = SimulatorMode.AUTOMATED,
        timeout_ms: int = 30000,
    ):
        self.agent = agent
        self.mode = mode
        self.timeout_ms = timeout_ms

        self._running = False
        self._current_scenario: Optional[TestScenario] = None
        self._conversation_history: List[TestMessage] = []

        # Hooks for extensibility
        self._before_step_hooks: List[Callable] = []
        self._after_step_hooks: List[Callable] = []
        self._on_assertion_hooks: List[Callable] = []

    def add_before_step_hook(
        self,
        hook: Callable[[TestStep], None],
    ) -> None:
        """Add a hook to run before each step."""
        self._before_step_hooks.append(hook)

    def add_after_step_hook(
        self,
        hook: Callable[[TestStep, Dict[str, Any]], None],
    ) -> None:
        """Add a hook to run after each step."""
        self._after_step_hooks.append(hook)

    async def execute_scenario(
        self,
        scenario: TestScenario,
    ) -> TestScenario:
        """
        Execute a test scenario and return results.

        Args:
            scenario: The test scenario to execute

        Returns:
            The scenario with updated status and results
        """
        self._current_scenario = scenario
        scenario.status = TestStatus.RUNNING
        scenario.started_at = datetime.utcnow()
        scenario.steps_passed = 0
        scenario.steps_failed = 0

        logger.info(f"Starting scenario: {scenario.name}")

        try:
            # Initialize agent with scenario config
            await self.agent.initialize(
                agent_id=scenario.agent_id,
                config=scenario.agent_config,
            )

            # Execute each step
            for i, step in enumerate(scenario.steps):
                step_start = time.time()
                step.started_at = datetime.utcnow()
                step.status = TestStatus.RUNNING

                # Run before-step hooks
                for hook in self._before_step_hooks:
                    await self._run_hook(hook, step)

                # Apply delay before step
                if step.delay_before_ms > 0:
                    await asyncio.sleep(step.delay_before_ms / 1000)

                try:
                    # Execute the step
                    result = await self._execute_step(step)

                    # Run after-step hooks
                    for hook in self._after_step_hooks:
                        await self._run_hook(hook, step, result)

                    # Evaluate assertions
                    all_passed = await self._evaluate_assertions(step, result)

                    # Update step status
                    step.agent_response = result.get("response", "")
                    step.actual_latency_ms = result.get("latency_ms", 0)
                    step.completed_at = datetime.utcnow()

                    if all_passed:
                        step.status = TestStatus.PASSED
                        scenario.steps_passed += 1
                    else:
                        step.status = TestStatus.FAILED
                        scenario.steps_failed += 1

                        if scenario.stop_on_first_failure:
                            logger.warning(
                                f"Stopping scenario due to failure at step {i + 1}"
                            )
                            break

                except asyncio.TimeoutError:
                    step.status = TestStatus.TIMEOUT
                    step.error_message = f"Step timed out after {step.timeout_ms}ms"
                    scenario.steps_failed += 1
                    if scenario.stop_on_first_failure:
                        break

                except Exception as e:
                    step.status = TestStatus.ERROR
                    step.error_message = str(e)
                    scenario.steps_failed += 1
                    logger.error(f"Step error: {e}")
                    if scenario.stop_on_first_failure:
                        break

                finally:
                    step.completed_at = datetime.utcnow()

                # Apply delay after step
                if step.delay_after_ms > 0:
                    await asyncio.sleep(step.delay_after_ms / 1000)

            # Determine final scenario status
            if scenario.steps_failed > 0:
                scenario.status = TestStatus.FAILED
            elif scenario.steps_passed == len(scenario.steps):
                scenario.status = TestStatus.PASSED
            else:
                scenario.status = TestStatus.SKIPPED

        except Exception as e:
            scenario.status = TestStatus.ERROR
            scenario.error_message = str(e)
            logger.error(f"Scenario error: {e}")

        finally:
            scenario.completed_at = datetime.utcnow()
            scenario.total_duration_ms = (
                (scenario.completed_at - scenario.started_at).total_seconds() * 1000
            )

            # Cleanup
            await self.agent.reset_conversation()
            self._current_scenario = None

        logger.info(
            f"Scenario completed: {scenario.name} - "
            f"Status: {scenario.status.value}, "
            f"Passed: {scenario.steps_passed}/{len(scenario.steps)}"
        )

        return scenario

    async def execute_suite(
        self,
        suite: TestSuite,
    ) -> TestReport:
        """
        Execute a complete test suite and generate report.

        Args:
            suite: The test suite to execute

        Returns:
            Comprehensive test report
        """
        report = TestReport(
            name=f"Test Report: {suite.name}",
            suite_id=suite.id,
            total_scenarios=len(suite.scenarios),
        )

        suite.status = TestStatus.RUNNING
        suite.started_at = datetime.utcnow()

        logger.info(f"Starting test suite: {suite.name}")

        try:
            if suite.parallel_execution:
                # Execute scenarios in parallel
                semaphore = asyncio.Semaphore(suite.max_parallel)

                async def run_with_semaphore(scenario):
                    async with semaphore:
                        return await self.execute_scenario(scenario)

                results = await asyncio.gather(
                    *[run_with_semaphore(s) for s in suite.scenarios],
                    return_exceptions=True,
                )

                for scenario, result in zip(suite.scenarios, results):
                    if isinstance(result, Exception):
                        scenario.status = TestStatus.ERROR
                        scenario.error_message = str(result)
            else:
                # Execute scenarios sequentially
                for scenario in suite.scenarios:
                    await self.execute_scenario(scenario)

                    if (suite.stop_on_first_failure and
                            scenario.status == TestStatus.FAILED):
                        # Mark remaining as skipped
                        idx = suite.scenarios.index(scenario)
                        for s in suite.scenarios[idx + 1:]:
                            s.status = TestStatus.SKIPPED
                        break

            # Compile results
            for scenario in suite.scenarios:
                if scenario.status == TestStatus.PASSED:
                    suite.scenarios_passed += 1
                    report.passed_scenarios += 1
                elif scenario.status == TestStatus.FAILED:
                    suite.scenarios_failed += 1
                    report.failed_scenarios += 1
                elif scenario.status == TestStatus.SKIPPED:
                    suite.scenarios_skipped += 1
                    report.skipped_scenarios += 1
                else:
                    report.error_scenarios += 1

                # Collect step statistics
                report.total_steps += len(scenario.steps)
                report.passed_steps += scenario.steps_passed
                report.failed_steps += scenario.steps_failed

                # Collect assertion details
                for step in scenario.steps:
                    for assertion in step.assertions:
                        report.total_assertions += 1
                        if assertion.passed:
                            report.passed_assertions += 1
                        else:
                            report.failed_assertions += 1
                            report.failed_assertions_details.append({
                                "scenario": scenario.name,
                                "step": step.name,
                                "assertion_type": assertion.assertion_type.value,
                                "expected": str(assertion.expected_value),
                                "actual": str(assertion.actual_value),
                                "message": assertion.message,
                            })

                    if step.error_message:
                        report.errors.append({
                            "scenario": scenario.name,
                            "step": step.name,
                            "error": step.error_message,
                        })

                # Add scenario to report
                report.scenario_results.append({
                    "id": scenario.id,
                    "name": scenario.name,
                    "status": scenario.status.value,
                    "steps_passed": scenario.steps_passed,
                    "steps_total": len(scenario.steps),
                    "duration_ms": scenario.total_duration_ms,
                })

            # Determine suite status
            if suite.scenarios_failed > 0 or report.error_scenarios > 0:
                suite.status = TestStatus.FAILED
            else:
                suite.status = TestStatus.PASSED

        finally:
            suite.completed_at = datetime.utcnow()
            suite.total_duration_ms = (
                (suite.completed_at - suite.started_at).total_seconds() * 1000
            )
            report.total_duration_ms = suite.total_duration_ms

            # Calculate averages
            if suite.scenarios_passed + suite.scenarios_failed > 0:
                report.avg_scenario_duration_ms = (
                    suite.total_duration_ms /
                    (suite.scenarios_passed + suite.scenarios_failed)
                )

            if report.total_steps > 0:
                total_latency = sum(
                    step.actual_latency_ms or 0
                    for scenario in suite.scenarios
                    for step in scenario.steps
                )
                report.avg_step_latency_ms = total_latency / report.total_steps

        logger.info(
            f"Suite completed: {suite.name} - "
            f"Passed: {suite.scenarios_passed}/{len(suite.scenarios)}"
        )

        return report

    async def replay_conversation(
        self,
        script: ConversationScript,
    ) -> TestReport:
        """
        Replay a recorded conversation and compare responses.

        Args:
            script: The conversation script to replay

        Returns:
            Test report with comparison results
        """
        # Create a scenario from the script
        scenario = TestScenario(
            name=f"Replay: {script.name}",
            description=f"Replaying conversation: {script.description}",
            category=TestCategory.REGRESSION,
            agent_id=script.agent_id,
        )

        # Extract user messages and create steps
        for i, msg in enumerate(script.messages):
            if msg.role == "user":
                # Find the next assistant message for expected response
                expected_response = None
                for next_msg in script.messages[i + 1:]:
                    if next_msg.role == "assistant":
                        expected_response = next_msg.content
                        break

                step = TestStep(
                    name=f"Turn {len(scenario.steps) + 1}",
                    user_input=msg.content,
                    user_audio=msg.audio_file,
                )

                # Add assertion for expected response if available
                if expected_response:
                    # Use contains assertion for flexibility
                    step.assertions.append(TestAssertion(
                        assertion_type=AssertionType.CONTAINS,
                        expected_value=expected_response[:50],  # First 50 chars
                    ))

                scenario.steps.append(step)

        # Execute the scenario
        await self.execute_scenario(scenario)

        # Generate report
        report = TestReport(
            name=f"Replay Report: {script.name}",
            total_scenarios=1,
        )

        if scenario.status == TestStatus.PASSED:
            report.passed_scenarios = 1
        else:
            report.failed_scenarios = 1

        report.total_steps = len(scenario.steps)
        report.passed_steps = scenario.steps_passed
        report.failed_steps = scenario.steps_failed

        return report

    async def _execute_step(
        self,
        step: TestStep,
    ) -> Dict[str, Any]:
        """Execute a single test step."""
        try:
            result = await asyncio.wait_for(
                self.agent.send_message(
                    step.user_input,
                    context=step.context,
                ),
                timeout=step.timeout_ms / 1000,
            )
            return result
        except asyncio.TimeoutError:
            raise TestTimeoutError(
                f"Step timed out after {step.timeout_ms}ms"
            )

    async def _evaluate_assertions(
        self,
        step: TestStep,
        result: Dict[str, Any],
    ) -> bool:
        """Evaluate all assertions for a step."""
        all_passed = True

        for assertion in step.assertions:
            # Determine what to evaluate based on assertion type
            if assertion.assertion_type in [
                AssertionType.CONTAINS,
                AssertionType.NOT_CONTAINS,
                AssertionType.EQUALS,
                AssertionType.MATCHES_REGEX,
            ]:
                actual = result.get("response", "")

            elif assertion.assertion_type == AssertionType.INTENT_MATCH:
                actual = result.get("metadata", {}).get("intent", {})

            elif assertion.assertion_type == AssertionType.SENTIMENT_RANGE:
                actual = result.get("metadata", {}).get("sentiment", 0)

            elif assertion.assertion_type == AssertionType.LATENCY_UNDER:
                actual = result.get("latency_ms", float('inf'))

            elif assertion.assertion_type == AssertionType.ENTITY_EXTRACTED:
                actual = result.get("metadata", {}).get("entities", [])

            elif assertion.assertion_type == AssertionType.TOOL_CALLED:
                actual = result.get("metadata", {}).get("tool_calls", [])

            elif assertion.assertion_type == AssertionType.TRANSFER_TRIGGERED:
                actual = result.get("metadata", {}).get("transfer", {})

            else:
                actual = result

            passed = assertion.evaluate(actual)

            # Run assertion hooks
            for hook in self._on_assertion_hooks:
                await self._run_hook(hook, assertion, passed)

            if not passed:
                all_passed = False
                logger.warning(
                    f"Assertion failed: {assertion.assertion_type.value} - "
                    f"{assertion.message}"
                )

        return all_passed

    async def _run_hook(
        self,
        hook: Callable,
        *args,
        **kwargs,
    ) -> None:
        """Run a hook function safely."""
        try:
            if asyncio.iscoroutinefunction(hook):
                await hook(*args, **kwargs)
            else:
                hook(*args, **kwargs)
        except Exception as e:
            logger.error(f"Hook error: {e}")


# =============================================================================
# Load Testing Simulator
# =============================================================================


class LoadTestSimulator:
    """
    Simulates concurrent load for performance testing.

    Creates multiple simulated users that interact with the agent
    simultaneously to test scalability and performance under load.
    """

    def __init__(
        self,
        agent_factory: Callable[[], AgentInterface],
        scenarios: List[TestScenario],
    ):
        self.agent_factory = agent_factory
        self.scenarios = scenarios
        self._running = False
        self._results: List[Dict[str, Any]] = []

    async def run_load_test(
        self,
        concurrent_users: int = 10,
        duration_seconds: int = 60,
        ramp_up_seconds: int = 10,
        scenario_selection: str = "random",  # random, round_robin, weighted
    ) -> BenchmarkResult:
        """
        Run a load test with multiple concurrent users.

        Args:
            concurrent_users: Number of simultaneous users
            duration_seconds: Total test duration
            ramp_up_seconds: Time to ramp up to full load
            scenario_selection: How to select scenarios for each user

        Returns:
            Benchmark results with performance metrics
        """
        result = BenchmarkResult(
            name=f"Load Test: {concurrent_users} users, {duration_seconds}s",
            concurrent_users=concurrent_users,
            duration_seconds=duration_seconds,
        )

        result.started_at = datetime.utcnow()
        self._running = True
        self._results.clear()

        # Initialize metrics
        result.metrics[BenchmarkMetric.RESPONSE_LATENCY] = LatencyMetrics()
        result.metrics[BenchmarkMetric.END_TO_END_LATENCY] = LatencyMetrics()

        logger.info(
            f"Starting load test: {concurrent_users} users, "
            f"{duration_seconds}s duration"
        )

        try:
            # Create user tasks with staggered start
            tasks = []
            for i in range(concurrent_users):
                delay = (ramp_up_seconds / concurrent_users) * i
                task = asyncio.create_task(
                    self._run_simulated_user(
                        user_id=i,
                        start_delay=delay,
                        duration=duration_seconds,
                        scenario_selection=scenario_selection,
                        result=result,
                    )
                )
                tasks.append(task)

            # Wait for all users to complete
            await asyncio.gather(*tasks, return_exceptions=True)

        finally:
            self._running = False
            result.completed_at = datetime.utcnow()

            # Calculate final statistics
            result.successful_requests = sum(
                1 for r in self._results if r.get("success", False)
            )
            result.failed_requests = sum(
                1 for r in self._results if not r.get("success", True)
            )

            total_requests = result.successful_requests + result.failed_requests
            if total_requests > 0:
                result.error_rate = result.failed_requests / total_requests
                result.requests_per_second = total_requests / duration_seconds

        logger.info(
            f"Load test completed: "
            f"{result.successful_requests} successful, "
            f"{result.failed_requests} failed, "
            f"{result.requests_per_second:.2f} req/s"
        )

        return result

    async def _run_simulated_user(
        self,
        user_id: int,
        start_delay: float,
        duration: int,
        scenario_selection: str,
        result: BenchmarkResult,
    ) -> None:
        """Run a single simulated user."""
        await asyncio.sleep(start_delay)

        agent = self.agent_factory()
        await agent.initialize()

        end_time = time.time() + duration - start_delay
        scenario_idx = user_id % len(self.scenarios)

        try:
            while time.time() < end_time and self._running:
                # Select scenario
                if scenario_selection == "random":
                    scenario = random.choice(self.scenarios)
                elif scenario_selection == "round_robin":
                    scenario = self.scenarios[scenario_idx]
                    scenario_idx = (scenario_idx + 1) % len(self.scenarios)
                else:
                    scenario = self.scenarios[0]

                # Execute a random step from the scenario
                if scenario.steps:
                    step = random.choice(scenario.steps)

                    request_start = time.time()
                    try:
                        response = await agent.send_message(step.user_input)

                        latency_ms = (time.time() - request_start) * 1000
                        result.metrics[BenchmarkMetric.RESPONSE_LATENCY].add_sample(
                            latency_ms
                        )

                        self._results.append({
                            "user_id": user_id,
                            "success": True,
                            "latency_ms": latency_ms,
                            "timestamp": datetime.utcnow(),
                        })

                    except Exception as e:
                        self._results.append({
                            "user_id": user_id,
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.utcnow(),
                        })

                # Random think time
                await asyncio.sleep(random.uniform(0.5, 2.0))

        finally:
            await agent.close()


# =============================================================================
# Test Builder (Fluent API)
# =============================================================================


class TestBuilder:
    """
    Fluent builder for creating test scenarios.

    Example:
        scenario = (
            TestBuilder("Order Flow Test")
            .with_priority(TestPriority.HIGH)
            .with_agent_config({"model": "gpt-4"})
            .add_step("Hello!")
                .expect_contains("Hello")
                .expect_latency_under(500)
            .add_step("I want to order a pizza")
                .expect_intent("order_food")
                .expect_entity("pizza")
            .add_step("Make it a large pepperoni")
                .expect_tool_call("create_order")
            .build()
        )
    """

    def __init__(
        self,
        name: str,
        description: str = "",
    ):
        self._scenario = TestScenario(
            name=name,
            description=description,
        )
        self._current_step: Optional[TestStep] = None

    def with_category(self, category: TestCategory) -> "TestBuilder":
        """Set the test category."""
        self._scenario.category = category
        return self

    def with_priority(self, priority: TestPriority) -> "TestBuilder":
        """Set the test priority."""
        self._scenario.priority = priority
        return self

    def with_tags(self, *tags: str) -> "TestBuilder":
        """Add tags to the scenario."""
        self._scenario.tags.update(tags)
        return self

    def with_agent_id(self, agent_id: str) -> "TestBuilder":
        """Set the agent ID."""
        self._scenario.agent_id = agent_id
        return self

    def with_agent_config(self, config: Dict[str, Any]) -> "TestBuilder":
        """Set agent configuration."""
        self._scenario.agent_config = config
        return self

    def with_initial_context(self, context: Dict[str, Any]) -> "TestBuilder":
        """Set initial conversation context."""
        self._scenario.initial_context = context
        return self

    def with_system_prompt(self, prompt: str) -> "TestBuilder":
        """Override the system prompt."""
        self._scenario.system_prompt_override = prompt
        return self

    def with_timeout(self, timeout_ms: int) -> "TestBuilder":
        """Set scenario timeout."""
        self._scenario.timeout_ms = timeout_ms
        return self

    def stop_on_failure(self, stop: bool = True) -> "TestBuilder":
        """Configure whether to stop on first failure."""
        self._scenario.stop_on_first_failure = stop
        return self

    def add_step(
        self,
        user_input: str,
        name: str = "",
        timeout_ms: int = 30000,
    ) -> "TestBuilder":
        """Add a new test step."""
        # Finalize previous step if any
        if self._current_step is not None:
            self._scenario.steps.append(self._current_step)

        self._current_step = TestStep(
            name=name or f"Step {len(self._scenario.steps) + 1}",
            user_input=user_input,
            timeout_ms=timeout_ms,
        )
        return self

    def with_audio(self, audio_file: str) -> "TestBuilder":
        """Add audio file to current step."""
        if self._current_step:
            self._current_step.user_audio = audio_file
        return self

    def with_context(self, context: Dict[str, Any]) -> "TestBuilder":
        """Add context to current step."""
        if self._current_step:
            self._current_step.context = context
        return self

    def with_delay_before(self, delay_ms: int) -> "TestBuilder":
        """Add delay before current step."""
        if self._current_step:
            self._current_step.delay_before_ms = delay_ms
        return self

    def with_delay_after(self, delay_ms: int) -> "TestBuilder":
        """Add delay after current step."""
        if self._current_step:
            self._current_step.delay_after_ms = delay_ms
        return self

    def expect_contains(self, text: str) -> "TestBuilder":
        """Assert response contains text."""
        if self._current_step:
            self._current_step.assertions.append(TestAssertion(
                assertion_type=AssertionType.CONTAINS,
                expected_value=text,
            ))
        return self

    def expect_not_contains(self, text: str) -> "TestBuilder":
        """Assert response doesn't contain text."""
        if self._current_step:
            self._current_step.assertions.append(TestAssertion(
                assertion_type=AssertionType.NOT_CONTAINS,
                expected_value=text,
            ))
        return self

    def expect_equals(self, text: str) -> "TestBuilder":
        """Assert response equals text."""
        if self._current_step:
            self._current_step.assertions.append(TestAssertion(
                assertion_type=AssertionType.EQUALS,
                expected_value=text,
            ))
        return self

    def expect_matches(self, regex: str) -> "TestBuilder":
        """Assert response matches regex."""
        if self._current_step:
            self._current_step.assertions.append(TestAssertion(
                assertion_type=AssertionType.MATCHES_REGEX,
                expected_value=regex,
            ))
        return self

    def expect_intent(self, intent: str) -> "TestBuilder":
        """Assert intent is detected."""
        if self._current_step:
            self._current_step.assertions.append(TestAssertion(
                assertion_type=AssertionType.INTENT_MATCH,
                expected_value=intent,
            ))
        return self

    def expect_sentiment(
        self,
        min_val: float,
        max_val: float,
    ) -> "TestBuilder":
        """Assert sentiment is within range."""
        if self._current_step:
            self._current_step.assertions.append(TestAssertion(
                assertion_type=AssertionType.SENTIMENT_RANGE,
                expected_value=(min_val, max_val),
            ))
        return self

    def expect_latency_under(self, latency_ms: int) -> "TestBuilder":
        """Assert response latency is under threshold."""
        if self._current_step:
            self._current_step.assertions.append(TestAssertion(
                assertion_type=AssertionType.LATENCY_UNDER,
                expected_value=latency_ms,
            ))
        return self

    def expect_entity(self, entity: str) -> "TestBuilder":
        """Assert entity is extracted."""
        if self._current_step:
            self._current_step.assertions.append(TestAssertion(
                assertion_type=AssertionType.ENTITY_EXTRACTED,
                expected_value=entity,
            ))
        return self

    def expect_tool_call(self, tool_name: str) -> "TestBuilder":
        """Assert tool is called."""
        if self._current_step:
            self._current_step.assertions.append(TestAssertion(
                assertion_type=AssertionType.TOOL_CALLED,
                expected_value=tool_name,
            ))
        return self

    def expect_transfer(
        self,
        target: Optional[str] = None,
    ) -> "TestBuilder":
        """Assert transfer is triggered."""
        if self._current_step:
            self._current_step.assertions.append(TestAssertion(
                assertion_type=AssertionType.TRANSFER_TRIGGERED,
                expected_value=target,
            ))
        return self

    def expect_custom(
        self,
        validator: Callable[[Any], bool],
    ) -> "TestBuilder":
        """Add custom assertion."""
        if self._current_step:
            self._current_step.assertions.append(TestAssertion(
                assertion_type=AssertionType.CUSTOM,
                custom_validator=validator,
            ))
        return self

    def build(self) -> TestScenario:
        """Build the test scenario."""
        # Add final step
        if self._current_step is not None:
            self._scenario.steps.append(self._current_step)

        return self._scenario


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Interfaces
    "AgentInterface",
    "MockAgentInterface",
    # Simulators
    "ConversationSimulator",
    "LoadTestSimulator",
    # Builder
    "TestBuilder",
]

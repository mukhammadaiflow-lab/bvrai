"""
Agent Testing & Simulation Framework - Base Types

Provides foundational types, enums, and dataclasses for the
comprehensive agent testing system.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# =============================================================================
# Enums
# =============================================================================


class TestStatus(str, Enum):
    """Status of a test execution."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class TestPriority(str, Enum):
    """Priority level for tests."""
    CRITICAL = "critical"  # Must pass for deployment
    HIGH = "high"          # Important but not blocking
    MEDIUM = "medium"      # Standard tests
    LOW = "low"            # Nice to have


class TestCategory(str, Enum):
    """Categories of tests."""
    UNIT = "unit"                    # Individual component tests
    INTEGRATION = "integration"      # Multi-component tests
    CONVERSATION = "conversation"    # Full conversation flow tests
    PERFORMANCE = "performance"      # Latency and throughput tests
    LOAD = "load"                    # Concurrent load tests
    REGRESSION = "regression"        # Regression detection tests
    VOICE_QUALITY = "voice_quality"  # Audio/voice quality tests
    COMPLIANCE = "compliance"        # Regulatory compliance tests


class AssertionType(str, Enum):
    """Types of assertions for test scenarios."""
    CONTAINS = "contains"                # Response contains text
    NOT_CONTAINS = "not_contains"        # Response doesn't contain text
    EQUALS = "equals"                    # Exact match
    MATCHES_REGEX = "matches_regex"      # Regex pattern match
    INTENT_MATCH = "intent_match"        # Intent classification match
    SENTIMENT_RANGE = "sentiment_range"  # Sentiment within range
    LATENCY_UNDER = "latency_under"      # Response time under threshold
    ENTITY_EXTRACTED = "entity_extracted"  # Named entity extracted
    TOOL_CALLED = "tool_called"          # Function/tool was invoked
    TRANSFER_TRIGGERED = "transfer_triggered"  # Call transfer triggered
    CUSTOM = "custom"                    # Custom assertion function


class SimulatorMode(str, Enum):
    """Modes for conversation simulation."""
    INTERACTIVE = "interactive"    # Step-by-step with user input
    AUTOMATED = "automated"        # Fully automated from script
    REPLAY = "replay"              # Replay recorded conversation
    RANDOM = "random"              # Random but valid inputs
    ADVERSARIAL = "adversarial"    # Edge cases and stress testing


class VoiceSimulationType(str, Enum):
    """Types of voice simulation."""
    TEXT_ONLY = "text_only"        # No audio, text-based simulation
    SYNTHESIZED = "synthesized"    # Use TTS for audio
    RECORDED = "recorded"          # Use pre-recorded audio
    LIVE = "live"                  # Live microphone input


class BenchmarkMetric(str, Enum):
    """Metrics for performance benchmarking."""
    RESPONSE_LATENCY = "response_latency"
    FIRST_TOKEN_LATENCY = "first_token_latency"
    TOKENS_PER_SECOND = "tokens_per_second"
    STT_LATENCY = "stt_latency"
    TTS_LATENCY = "tts_latency"
    END_TO_END_LATENCY = "end_to_end_latency"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CONCURRENT_CAPACITY = "concurrent_capacity"
    ERROR_RATE = "error_rate"


# =============================================================================
# Core Data Classes
# =============================================================================


@dataclass
class TestAssertion:
    """A single assertion to validate test results."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assertion_type: AssertionType = AssertionType.CONTAINS
    expected_value: Any = None
    actual_value: Any = None
    passed: bool = False
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For custom assertions
    custom_validator: Optional[Callable[[Any], bool]] = None

    def evaluate(self, actual: Any) -> bool:
        """Evaluate the assertion against actual value."""
        self.actual_value = actual

        try:
            if self.assertion_type == AssertionType.CONTAINS:
                self.passed = self.expected_value in str(actual)
                self.message = (
                    f"Expected '{actual}' to contain '{self.expected_value}'"
                    if not self.passed else "Contains check passed"
                )

            elif self.assertion_type == AssertionType.NOT_CONTAINS:
                self.passed = self.expected_value not in str(actual)
                self.message = (
                    f"Expected '{actual}' to not contain '{self.expected_value}'"
                    if not self.passed else "Not contains check passed"
                )

            elif self.assertion_type == AssertionType.EQUALS:
                self.passed = actual == self.expected_value
                self.message = (
                    f"Expected '{self.expected_value}', got '{actual}'"
                    if not self.passed else "Equals check passed"
                )

            elif self.assertion_type == AssertionType.MATCHES_REGEX:
                import re
                pattern = re.compile(self.expected_value)
                self.passed = bool(pattern.search(str(actual)))
                self.message = (
                    f"Pattern '{self.expected_value}' not found in '{actual}'"
                    if not self.passed else "Regex match passed"
                )

            elif self.assertion_type == AssertionType.INTENT_MATCH:
                # actual should be {"intent": "...", "confidence": ...}
                if isinstance(actual, dict):
                    self.passed = actual.get("intent") == self.expected_value
                else:
                    self.passed = False
                self.message = (
                    f"Expected intent '{self.expected_value}', got '{actual}'"
                    if not self.passed else "Intent match passed"
                )

            elif self.assertion_type == AssertionType.SENTIMENT_RANGE:
                # expected_value should be (min, max) tuple
                min_val, max_val = self.expected_value
                if isinstance(actual, (int, float)):
                    self.passed = min_val <= actual <= max_val
                else:
                    self.passed = False
                self.message = (
                    f"Sentiment {actual} not in range [{min_val}, {max_val}]"
                    if not self.passed else "Sentiment range check passed"
                )

            elif self.assertion_type == AssertionType.LATENCY_UNDER:
                if isinstance(actual, (int, float)):
                    self.passed = actual < self.expected_value
                else:
                    self.passed = False
                self.message = (
                    f"Latency {actual}ms exceeds threshold {self.expected_value}ms"
                    if not self.passed else "Latency check passed"
                )

            elif self.assertion_type == AssertionType.ENTITY_EXTRACTED:
                # actual should be list of entities
                if isinstance(actual, list):
                    entity_texts = [e.get("text", e) for e in actual]
                    self.passed = self.expected_value in entity_texts
                else:
                    self.passed = False
                self.message = (
                    f"Entity '{self.expected_value}' not found in {actual}"
                    if not self.passed else "Entity extraction check passed"
                )

            elif self.assertion_type == AssertionType.TOOL_CALLED:
                # actual should be list of tool calls
                if isinstance(actual, list):
                    tool_names = [t.get("name", t) for t in actual]
                    self.passed = self.expected_value in tool_names
                else:
                    self.passed = False
                self.message = (
                    f"Tool '{self.expected_value}' not called"
                    if not self.passed else "Tool call check passed"
                )

            elif self.assertion_type == AssertionType.TRANSFER_TRIGGERED:
                if isinstance(actual, dict):
                    self.passed = actual.get("transfer_triggered", False)
                    if self.expected_value:
                        self.passed = self.passed and (
                            actual.get("transfer_target") == self.expected_value
                        )
                else:
                    self.passed = bool(actual)
                self.message = (
                    "Transfer not triggered as expected"
                    if not self.passed else "Transfer trigger check passed"
                )

            elif self.assertion_type == AssertionType.CUSTOM:
                if self.custom_validator:
                    self.passed = self.custom_validator(actual)
                    self.message = (
                        "Custom validation failed"
                        if not self.passed else "Custom validation passed"
                    )
                else:
                    self.passed = False
                    self.message = "No custom validator provided"

            else:
                self.passed = False
                self.message = f"Unknown assertion type: {self.assertion_type}"

        except Exception as e:
            self.passed = False
            self.message = f"Assertion error: {str(e)}"

        return self.passed


@dataclass
class TestMessage:
    """A message in a test conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    audio_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timing information
    latency_ms: Optional[float] = None
    processing_time_ms: Optional[float] = None


@dataclass
class TestStep:
    """A single step in a test scenario."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Input
    user_input: str = ""
    user_audio: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    # Expected output
    assertions: List[TestAssertion] = field(default_factory=list)

    # Timing
    timeout_ms: int = 30000
    delay_before_ms: int = 0
    delay_after_ms: int = 0

    # Results (populated after execution)
    status: TestStatus = TestStatus.PENDING
    agent_response: Optional[str] = None
    actual_latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class TestScenario:
    """A complete test scenario with multiple steps."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: TestCategory = TestCategory.CONVERSATION
    priority: TestPriority = TestPriority.MEDIUM
    tags: Set[str] = field(default_factory=set)

    # Agent configuration
    agent_id: Optional[str] = None
    agent_config: Dict[str, Any] = field(default_factory=dict)

    # Initial context
    initial_context: Dict[str, Any] = field(default_factory=dict)
    system_prompt_override: Optional[str] = None

    # Test steps
    steps: List[TestStep] = field(default_factory=list)

    # Execution settings
    timeout_ms: int = 120000
    retry_count: int = 0
    stop_on_first_failure: bool = True

    # Results (populated after execution)
    status: TestStatus = TestStatus.PENDING
    steps_passed: int = 0
    steps_failed: int = 0
    total_duration_ms: float = 0.0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def add_step(
        self,
        user_input: str,
        name: str = "",
        assertions: Optional[List[TestAssertion]] = None,
        **kwargs,
    ) -> TestStep:
        """Add a step to the scenario."""
        step = TestStep(
            name=name or f"Step {len(self.steps) + 1}",
            user_input=user_input,
            assertions=assertions or [],
            **kwargs,
        )
        self.steps.append(step)
        return step

    def add_assertion(
        self,
        step_index: int,
        assertion_type: AssertionType,
        expected_value: Any,
    ) -> TestAssertion:
        """Add an assertion to a specific step."""
        if 0 <= step_index < len(self.steps):
            assertion = TestAssertion(
                assertion_type=assertion_type,
                expected_value=expected_value,
            )
            self.steps[step_index].assertions.append(assertion)
            return assertion
        raise IndexError(f"Step index {step_index} out of range")


@dataclass
class TestSuite:
    """A collection of test scenarios."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"

    # Organization
    organization_id: Optional[str] = None
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Scenarios
    scenarios: List[TestScenario] = field(default_factory=list)

    # Execution settings
    parallel_execution: bool = False
    max_parallel: int = 5
    stop_on_first_failure: bool = False

    # Results (populated after execution)
    status: TestStatus = TestStatus.PENDING
    scenarios_passed: int = 0
    scenarios_failed: int = 0
    scenarios_skipped: int = 0
    total_duration_ms: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def add_scenario(self, scenario: TestScenario) -> None:
        """Add a scenario to the suite."""
        self.scenarios.append(scenario)

    def get_scenarios_by_category(
        self,
        category: TestCategory,
    ) -> List[TestScenario]:
        """Get scenarios by category."""
        return [s for s in self.scenarios if s.category == category]

    def get_scenarios_by_priority(
        self,
        priority: TestPriority,
    ) -> List[TestScenario]:
        """Get scenarios by priority."""
        return [s for s in self.scenarios if s.priority == priority]


# =============================================================================
# Performance & Benchmark Types
# =============================================================================


@dataclass
class LatencyMetrics:
    """Latency statistics for a set of measurements."""
    count: int = 0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    std_dev_ms: float = 0.0

    _samples: List[float] = field(default_factory=list, repr=False)

    def add_sample(self, latency_ms: float) -> None:
        """Add a latency sample."""
        self._samples.append(latency_ms)
        self._recalculate()

    def _recalculate(self) -> None:
        """Recalculate statistics."""
        if not self._samples:
            return

        import statistics

        self.count = len(self._samples)
        self.min_ms = min(self._samples)
        self.max_ms = max(self._samples)
        self.mean_ms = statistics.mean(self._samples)

        sorted_samples = sorted(self._samples)
        self.median_ms = statistics.median(sorted_samples)

        # Percentiles
        p95_idx = int(len(sorted_samples) * 0.95)
        p99_idx = int(len(sorted_samples) * 0.99)
        self.p95_ms = sorted_samples[min(p95_idx, len(sorted_samples) - 1)]
        self.p99_ms = sorted_samples[min(p99_idx, len(sorted_samples) - 1)]

        if len(self._samples) > 1:
            self.std_dev_ms = statistics.stdev(self._samples)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
        }


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Configuration
    agent_id: Optional[str] = None
    concurrent_users: int = 1
    duration_seconds: int = 60
    warmup_seconds: int = 5

    # Metrics
    metrics: Dict[BenchmarkMetric, LatencyMetrics] = field(default_factory=dict)
    requests_per_second: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Resource usage
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0

    def get_metric(self, metric: BenchmarkMetric) -> Optional[LatencyMetrics]:
        """Get a specific metric."""
        return self.metrics.get(metric)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_id": self.agent_id,
            "concurrent_users": self.concurrent_users,
            "duration_seconds": self.duration_seconds,
            "requests_per_second": round(self.requests_per_second, 2),
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": round(self.error_rate * 100, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "avg_cpu_percent": round(self.avg_cpu_percent, 2),
            "metrics": {
                k.value: v.to_dict()
                for k, v in self.metrics.items()
            },
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# =============================================================================
# Simulation Types
# =============================================================================


@dataclass
class SimulatedUser:
    """Configuration for a simulated user in load testing."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Simulated User"
    persona: str = "default"

    # Behavior settings
    think_time_ms: Tuple[int, int] = (1000, 3000)  # min, max
    typing_speed_cpm: int = 200  # characters per minute
    patience_ms: int = 30000  # max wait for response

    # Conversation settings
    max_turns: int = 10
    conversation_style: str = "normal"  # normal, verbose, terse

    # Voice settings (for voice simulation)
    voice_type: VoiceSimulationType = VoiceSimulationType.TEXT_ONLY
    audio_quality: str = "high"

    # State
    current_session_id: Optional[str] = None
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0


@dataclass
class ConversationScript:
    """A scripted conversation for replay or automation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Messages
    messages: List[TestMessage] = field(default_factory=list)

    # Metadata
    source: str = ""  # "recorded", "generated", "manual"
    recorded_at: Optional[datetime] = None
    agent_id: Optional[str] = None
    duration_ms: float = 0.0

    # Tags for filtering
    tags: Set[str] = field(default_factory=set)
    intent_labels: Set[str] = field(default_factory=set)

    def add_message(
        self,
        role: str,
        content: str,
        audio_file: Optional[str] = None,
    ) -> TestMessage:
        """Add a message to the script."""
        message = TestMessage(
            role=role,
            content=content,
            audio_file=audio_file,
        )
        self.messages.append(message)
        return message


# =============================================================================
# Test Report Types
# =============================================================================


@dataclass
class TestReport:
    """Comprehensive test execution report."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Execution context
    suite_id: Optional[str] = None
    environment: str = "test"  # test, staging, production
    agent_version: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Summary
    total_scenarios: int = 0
    passed_scenarios: int = 0
    failed_scenarios: int = 0
    skipped_scenarios: int = 0
    error_scenarios: int = 0

    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0

    total_assertions: int = 0
    passed_assertions: int = 0
    failed_assertions: int = 0

    # Timing
    total_duration_ms: float = 0.0
    avg_scenario_duration_ms: float = 0.0
    avg_step_latency_ms: float = 0.0

    # Details
    scenario_results: List[Dict[str, Any]] = field(default_factory=list)
    failed_assertions_details: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Coverage
    intents_tested: Set[str] = field(default_factory=set)
    entities_tested: Set[str] = field(default_factory=set)
    tools_tested: Set[str] = field(default_factory=set)

    @property
    def pass_rate(self) -> float:
        """Calculate overall pass rate."""
        total = self.passed_scenarios + self.failed_scenarios + self.error_scenarios
        if total == 0:
            return 0.0
        return self.passed_scenarios / total

    @property
    def is_passing(self) -> bool:
        """Check if all tests passed."""
        return self.failed_scenarios == 0 and self.error_scenarios == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "environment": self.environment,
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_scenarios": self.total_scenarios,
                "passed": self.passed_scenarios,
                "failed": self.failed_scenarios,
                "skipped": self.skipped_scenarios,
                "errors": self.error_scenarios,
                "pass_rate": f"{self.pass_rate * 100:.1f}%",
            },
            "timing": {
                "total_duration_ms": round(self.total_duration_ms, 2),
                "avg_scenario_duration_ms": round(self.avg_scenario_duration_ms, 2),
                "avg_step_latency_ms": round(self.avg_step_latency_ms, 2),
            },
            "coverage": {
                "intents_tested": len(self.intents_tested),
                "entities_tested": len(self.entities_tested),
                "tools_tested": len(self.tools_tested),
            },
            "failed_assertions": self.failed_assertions_details,
            "errors": self.errors,
        }

    def generate_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Test Report: {self.name}",
            "",
            f"**Environment:** {self.environment}",
            f"**Timestamp:** {self.timestamp.isoformat()}",
            f"**Agent Version:** {self.agent_version or 'N/A'}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Scenarios | {self.total_scenarios} |",
            f"| Passed | {self.passed_scenarios} |",
            f"| Failed | {self.failed_scenarios} |",
            f"| Skipped | {self.skipped_scenarios} |",
            f"| Errors | {self.error_scenarios} |",
            f"| Pass Rate | {self.pass_rate * 100:.1f}% |",
            "",
            "## Timing",
            "",
            f"- Total Duration: {self.total_duration_ms / 1000:.2f}s",
            f"- Avg Scenario: {self.avg_scenario_duration_ms:.2f}ms",
            f"- Avg Step Latency: {self.avg_step_latency_ms:.2f}ms",
            "",
        ]

        if self.failed_assertions_details:
            lines.extend([
                "## Failed Assertions",
                "",
            ])
            for detail in self.failed_assertions_details:
                lines.append(f"- **{detail.get('scenario', 'Unknown')}**: "
                             f"{detail.get('message', 'No message')}")
            lines.append("")

        if self.errors:
            lines.extend([
                "## Errors",
                "",
            ])
            for error in self.errors:
                lines.append(f"- **{error.get('scenario', 'Unknown')}**: "
                             f"{error.get('error', 'No error message')}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Exceptions
# =============================================================================


class TestError(Exception):
    """Base exception for testing errors."""
    pass


class TestTimeoutError(TestError):
    """Test exceeded timeout."""
    pass


class TestAssertionError(TestError):
    """Test assertion failed."""
    pass


class TestSetupError(TestError):
    """Test setup failed."""
    pass


class TestTeardownError(TestError):
    """Test teardown failed."""
    pass


class SimulationError(TestError):
    """Simulation error."""
    pass


class BenchmarkError(TestError):
    """Benchmark error."""
    pass


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "TestStatus",
    "TestPriority",
    "TestCategory",
    "AssertionType",
    "SimulatorMode",
    "VoiceSimulationType",
    "BenchmarkMetric",
    # Core types
    "TestAssertion",
    "TestMessage",
    "TestStep",
    "TestScenario",
    "TestSuite",
    # Performance types
    "LatencyMetrics",
    "BenchmarkResult",
    # Simulation types
    "SimulatedUser",
    "ConversationScript",
    # Report types
    "TestReport",
    # Exceptions
    "TestError",
    "TestTimeoutError",
    "TestAssertionError",
    "TestSetupError",
    "TestTeardownError",
    "SimulationError",
    "BenchmarkError",
]

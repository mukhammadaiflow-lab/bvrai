"""
Agent Testing & Simulation Framework

This module provides comprehensive testing capabilities for AI voice agents,
enabling quality assurance, regression testing, performance benchmarking,
and load testing before deployment to production.

Key Features:
- Conversation simulation with configurable modes
- Scenario-based testing with rich assertions
- Performance benchmarking with detailed metrics
- Load testing with concurrent simulated users
- Fluent test builder API
- Comprehensive test reports

Example Usage:

    from platform.testing import (
        TestBuilder,
        ConversationSimulator,
        LoadTestSimulator,
        MockAgentInterface,
        TestSuite,
        TestPriority,
        TestCategory,
    )

    # =================================================================
    # 1. Build a Test Scenario using Fluent API
    # =================================================================

    scenario = (
        TestBuilder("Customer Support Flow")
        .with_priority(TestPriority.HIGH)
        .with_category(TestCategory.CONVERSATION)
        .with_tags("support", "ordering", "critical-path")
        .with_agent_config({"model": "gpt-4"})

        # Step 1: Greeting
        .add_step("Hello, I need help with my order")
            .expect_contains("help")
            .expect_intent("support_request")
            .expect_latency_under(500)

        # Step 2: Order inquiry
        .add_step("Order number is ORD-12345")
            .expect_contains("ORD-12345")
            .expect_entity("ORD-12345")
            .expect_tool_call("lookup_order")

        # Step 3: Issue description
        .add_step("The item arrived damaged")
            .expect_contains("sorry")
            .expect_sentiment(-0.8, -0.2)
            .expect_intent("complaint")

        # Step 4: Resolution
        .add_step("Yes, please send a replacement")
            .expect_tool_call("create_replacement_order")
            .expect_contains("replacement")

        .build()
    )

    # =================================================================
    # 2. Run Individual Scenario
    # =================================================================

    # Create agent interface (use MockAgentInterface for testing)
    agent = MockAgentInterface(
        responses={
            "hello": "Hello! How can I help you today?",
            "order": "I found order ORD-12345. What seems to be the issue?",
            "damaged": "I'm sorry to hear that. Would you like a replacement?",
            "replacement": "Perfect, I'll create a replacement order for you.",
        },
        latency_range=(100, 300),
    )

    # Create simulator
    simulator = ConversationSimulator(agent)

    # Execute scenario
    result = await simulator.execute_scenario(scenario)

    print(f"Status: {result.status.value}")
    print(f"Steps passed: {result.steps_passed}/{len(result.steps)}")

    # =================================================================
    # 3. Run Test Suite with Multiple Scenarios
    # =================================================================

    # Create test suite
    suite = TestSuite(
        name="Customer Support Test Suite",
        description="Comprehensive tests for customer support agent",
    )

    # Add scenarios
    suite.add_scenario(scenario)
    suite.add_scenario(another_scenario)

    # Execute suite
    report = await simulator.execute_suite(suite)

    # Print results
    print(report.generate_markdown())

    # =================================================================
    # 4. Load Testing
    # =================================================================

    # Create load test simulator
    load_simulator = LoadTestSimulator(
        agent_factory=lambda: MockAgentInterface(),
        scenarios=[scenario],
    )

    # Run load test
    benchmark = await load_simulator.run_load_test(
        concurrent_users=50,
        duration_seconds=120,
        ramp_up_seconds=30,
    )

    print(f"Requests/second: {benchmark.requests_per_second:.2f}")
    print(f"P95 latency: {benchmark.metrics['response_latency'].p95_ms:.2f}ms")
    print(f"Error rate: {benchmark.error_rate * 100:.2f}%")

    # =================================================================
    # 5. Replay Recorded Conversations
    # =================================================================

    from platform.testing import ConversationScript

    # Create script from recorded conversation
    script = ConversationScript(
        name="Real Customer Conversation",
        source="recorded",
    )
    script.add_message("user", "Hi, I need help")
    script.add_message("assistant", "Hello! How can I assist you today?")
    script.add_message("user", "I want to cancel my subscription")
    script.add_message("assistant", "I understand. Let me help with that.")

    # Replay and compare
    report = await simulator.replay_conversation(script)
    print(f"Replay match rate: {report.pass_rate * 100:.1f}%")

    # =================================================================
    # 6. Custom Assertions
    # =================================================================

    def validate_json_response(response):
        \"\"\"Custom validator for JSON responses.\"\"\"
        import json
        try:
            data = json.loads(response)
            return "status" in data and data["status"] == "success"
        except:
            return False

    scenario = (
        TestBuilder("API Response Test")
        .add_step("Get order status")
            .expect_custom(validate_json_response)
        .build()
    )

Test Categories:
    - UNIT: Individual component tests
    - INTEGRATION: Multi-component integration tests
    - CONVERSATION: Full conversation flow tests
    - PERFORMANCE: Latency and throughput tests
    - LOAD: Concurrent load tests
    - REGRESSION: Regression detection tests
    - VOICE_QUALITY: Audio/voice quality tests
    - COMPLIANCE: Regulatory compliance tests

Assertion Types:
    - CONTAINS: Response contains expected text
    - NOT_CONTAINS: Response doesn't contain text
    - EQUALS: Exact match
    - MATCHES_REGEX: Regex pattern match
    - INTENT_MATCH: Intent classification match
    - SENTIMENT_RANGE: Sentiment score within range
    - LATENCY_UNDER: Response time threshold
    - ENTITY_EXTRACTED: Named entity extraction
    - TOOL_CALLED: Function/tool invocation
    - TRANSFER_TRIGGERED: Call transfer detection
    - CUSTOM: Custom validation function

Benchmark Metrics:
    - RESPONSE_LATENCY: End-to-end response time
    - FIRST_TOKEN_LATENCY: Time to first token
    - TOKENS_PER_SECOND: Throughput metric
    - STT_LATENCY: Speech-to-text processing time
    - TTS_LATENCY: Text-to-speech processing time
    - END_TO_END_LATENCY: Full pipeline latency
    - MEMORY_USAGE: Peak memory consumption
    - CPU_USAGE: Average CPU utilization
    - CONCURRENT_CAPACITY: Max concurrent users
    - ERROR_RATE: Failure rate under load
"""

# Import all base types
from .base import (
    # Enums
    TestStatus,
    TestPriority,
    TestCategory,
    AssertionType,
    SimulatorMode,
    VoiceSimulationType,
    BenchmarkMetric,
    # Core types
    TestAssertion,
    TestMessage,
    TestStep,
    TestScenario,
    TestSuite,
    # Performance types
    LatencyMetrics,
    BenchmarkResult,
    # Simulation types
    SimulatedUser,
    ConversationScript,
    # Report types
    TestReport,
    # Exceptions
    TestError,
    TestTimeoutError,
    TestAssertionError,
    TestSetupError,
    TestTeardownError,
    SimulationError,
    BenchmarkError,
)

# Import simulator components
from .simulator import (
    # Interfaces
    AgentInterface,
    MockAgentInterface,
    # Simulators
    ConversationSimulator,
    LoadTestSimulator,
    # Builder
    TestBuilder,
)


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
    # Interfaces
    "AgentInterface",
    "MockAgentInterface",
    # Simulators
    "ConversationSimulator",
    "LoadTestSimulator",
    # Builder
    "TestBuilder",
]

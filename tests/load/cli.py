#!/usr/bin/env python3
"""CLI for running load tests."""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .base import LoadTestConfig, LoadTestResult
from .runner import LoadTestRunner, LoadTestSuite, run_quick_test, run_stress_test, run_soak_test
from .report import LoadTestReport, save_report
from .scenarios import (
    APILoadTest,
    CallSimulationTest,
    WebSocketLoadTest,
    ConcurrentCallsTest,
    DatabaseLoadTest,
    MixedWorkloadTest,
)


def load_config(config_path: str) -> dict:
    """Load configuration from file."""
    path = Path(config_path)

    if not path.exists():
        return {}

    with open(path) as f:
        if path.suffix in ('.yaml', '.yml'):
            if not YAML_AVAILABLE:
                print("ERROR: PyYAML is required to load YAML config files")
                print("Install with: pip install pyyaml")
                sys.exit(1)
            return yaml.safe_load(f)
        else:
            return json.load(f)


def expand_env_vars(value: str) -> str:
    """Expand environment variables in string."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var, "")
    return value


def create_config(args, file_config: dict) -> LoadTestConfig:
    """Create LoadTestConfig from args and file config."""
    # Get profile from file config
    profile = {}
    if args.profile and file_config.get("profiles"):
        profile = file_config["profiles"].get(args.profile, {})

    # Get target config
    target = file_config.get("target", {})

    # Get thresholds
    thresholds = file_config.get("thresholds", {})

    # Build config with priority: CLI args > profile > file config > defaults
    return LoadTestConfig(
        base_url=args.base_url or expand_env_vars(target.get("base_url", "")) or "http://localhost:8000",
        api_key=args.api_key or expand_env_vars(target.get("api_key", "")) or os.environ.get("BE_API_KEY", ""),
        organization_id=args.org_id or expand_env_vars(target.get("organization_id", "")) or os.environ.get("BE_ORG_ID", ""),
        websocket_url=expand_env_vars(target.get("websocket_url", "")),
        database_url=expand_env_vars(target.get("database_url", "")),

        concurrent_users=args.users or profile.get("concurrent_users", 10),
        requests_per_second=args.rps or profile.get("requests_per_second", 100),
        duration_seconds=args.duration or profile.get("duration_seconds", 60),
        ramp_up_seconds=args.ramp_up or profile.get("ramp_up_seconds", 10),
        ramp_down_seconds=profile.get("ramp_down_seconds", 5),

        max_avg_response_time_ms=thresholds.get("max_avg_response_time_ms", 500),
        max_p95_response_time_ms=thresholds.get("max_p95_response_time_ms", 1000),
        max_p99_response_time_ms=thresholds.get("max_p99_response_time_ms", 2000),
        max_error_rate=thresholds.get("max_error_rate", 0.01),
        min_requests_per_second=thresholds.get("min_requests_per_second", 50),

        output_dir=args.output or file_config.get("output", {}).get("directory", "./load_test_results"),
        save_detailed_results=file_config.get("output", {}).get("save_detailed_results", True),
    )


def progress_callback(update: dict) -> None:
    """Print progress updates."""
    update_type = update.get("type", "")

    if update_type == "test_starting":
        print(f"\n{'='*60}")
        print(f"Starting test {update['test_number']}/{update['total_tests']}: {update['test_name']}")
        print(f"{'='*60}")

    elif update_type == "progress":
        requests = update.get("total_requests", 0)
        errors = update.get("failed_requests", 0)
        avg_time = update.get("avg_response_time", 0)
        print(f"\r  Requests: {requests:,} | Errors: {errors:,} | Avg Response: {avg_time:.0f}ms", end="")

    elif update_type == "test_completed":
        print(f"\n  Completed: {update['test_name']}")
        print(f"  Status: {'PASSED' if update['passed'] else 'FAILED'}")
        print(f"  Duration: {update['duration']:.1f}s | Requests: {update['requests']:,} | Error Rate: {update['error_rate']:.2%}")


async def run_tests(args, config: LoadTestConfig) -> int:
    """Run the specified tests."""
    # Determine which tests to run
    test_classes = []

    if args.test == "all":
        test_classes = [
            APILoadTest,
            CallSimulationTest,
            WebSocketLoadTest,
            ConcurrentCallsTest,
        ]
    elif args.test == "api":
        test_classes = [APILoadTest]
    elif args.test == "calls":
        test_classes = [CallSimulationTest]
    elif args.test == "websocket":
        test_classes = [WebSocketLoadTest]
    elif args.test == "concurrent":
        test_classes = [ConcurrentCallsTest]
    elif args.test == "database":
        test_classes = [DatabaseLoadTest]
    elif args.test == "mixed":
        test_classes = [MixedWorkloadTest]
    elif args.test == "quick":
        print("Running quick smoke test...")
        result = await run_quick_test(config)
        return 0 if result.passed else 1
    elif args.test == "stress":
        print("Running stress test...")
        result = await run_stress_test(config)
        return 0 if result.passed else 1
    elif args.test == "soak":
        print(f"Running soak test for {args.duration // 3600} hours...")
        result = await run_soak_test(config, args.duration // 3600 or 1)
        return 0 if result.passed else 1

    # Create and run test suite
    runner = LoadTestRunner(config, output_dir=config.output_dir)
    runner.add_tests(test_classes)

    print(f"\nLoad Test Configuration:")
    print(f"  Target: {config.base_url}")
    print(f"  Concurrent Users: {config.concurrent_users}")
    print(f"  Target RPS: {config.requests_per_second}")
    print(f"  Duration: {config.duration_seconds}s")
    print(f"  Tests: {', '.join(t.__name__ for t in test_classes)}")
    print()

    # Run tests
    results = await runner.run_all(
        parallel=args.parallel,
        progress_callback=progress_callback if not args.quiet else None,
    )

    # Generate report
    report = LoadTestReport(
        title=f"Load Test Report - {datetime.utcnow().strftime('%Y-%m-%d')}",
        results=results,
    )
    report.analyze()

    # Save reports
    formats = ["json", "html"]
    if args.markdown:
        formats.append("markdown")

    saved_files = save_report(report, config.output_dir, formats=formats)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Status: {'PASSED' if report.overall_passed else 'FAILED'}")
    print(f"Tests Passed: {sum(1 for r in results if r.passed)}/{len(results)}")
    print(f"Total Requests: {sum(r.total_requests for r in results):,}")
    print(f"Total Errors: {sum(r.failed_requests for r in results):,}")

    if report.critical_issues:
        print(f"\nCritical Issues:")
        for issue in report.critical_issues:
            print(f"  - {issue}")

    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")

    print(f"\nReports saved to:")
    for fmt, path in saved_files.items():
        print(f"  {fmt}: {path}")

    return 0 if report.overall_passed else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Builder Engine Load Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test
  python -m tests.load.cli --test quick

  # Run all tests with default settings
  python -m tests.load.cli --test all

  # Run API tests with custom configuration
  python -m tests.load.cli --test api --users 50 --rps 200 --duration 120

  # Run tests using a profile from config file
  python -m tests.load.cli --config config.yaml --profile heavy --test all

  # Run stress test
  python -m tests.load.cli --test stress

  # Run 1-hour soak test
  python -m tests.load.cli --test soak --duration 3600
        """,
    )

    # Target options
    parser.add_argument("--base-url", "-u", help="Target base URL")
    parser.add_argument("--api-key", "-k", help="API key for authentication")
    parser.add_argument("--org-id", "-o", help="Organization ID")

    # Test selection
    parser.add_argument(
        "--test", "-t",
        choices=["all", "api", "calls", "websocket", "concurrent", "database", "mixed", "quick", "stress", "soak"],
        default="all",
        help="Test type to run",
    )

    # Load configuration
    parser.add_argument("--users", "-n", type=int, help="Number of concurrent users")
    parser.add_argument("--rps", "-r", type=int, help="Target requests per second")
    parser.add_argument("--duration", "-d", type=int, help="Test duration in seconds")
    parser.add_argument("--ramp-up", type=int, help="Ramp-up time in seconds")

    # Config file
    parser.add_argument("--config", "-c", default="./tests/load/config.yaml", help="Config file path")
    parser.add_argument("--profile", "-p", help="Profile name from config file")

    # Output options
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--markdown", action="store_true", help="Generate Markdown report")

    # Execution options
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Load config file
    file_config = load_config(args.config) if os.path.exists(args.config) else {}

    # Create config
    config = create_config(args, file_config)

    # Run tests
    try:
        exit_code = asyncio.run(run_tests(args, config))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Load Testing Suite

Comprehensive load testing framework for Builder Engine platform.

## Features

- **Multiple Test Scenarios**: API, WebSocket, Call Simulation, Database
- **Configurable Load Profiles**: Smoke, Light, Normal, Heavy, Stress, Soak
- **Rich Reporting**: JSON, HTML, and Markdown reports with charts
- **Threshold-Based Pass/Fail**: Configure SLOs for automated CI/CD
- **Real-time Progress**: Monitor test execution in real-time
- **Concurrent Testing**: Run multiple test types in parallel

## Installation

```bash
# Install dependencies
pip install aiohttp websockets asyncpg pyyaml

# Or with the project
pip install -e ".[load-testing]"
```

## Quick Start

### Run a Quick Smoke Test

```bash
python -m tests.load.cli --test quick --base-url http://localhost:8000
```

### Run All Tests with Default Settings

```bash
export BE_API_KEY="your-api-key"
export BE_ORG_ID="your-org-id"

python -m tests.load.cli --test all
```

### Run with Custom Configuration

```bash
python -m tests.load.cli \
  --test api \
  --base-url http://api.example.com \
  --users 50 \
  --rps 200 \
  --duration 300
```

### Run Using a Profile

```bash
python -m tests.load.cli --config config.yaml --profile heavy --test all
```

## Test Scenarios

### API Load Test (`api`)
Tests REST API endpoints with realistic request distribution:
- Agent management endpoints
- Call management endpoints
- Analytics queries
- Health checks

### Call Simulation Test (`calls`)
Simulates concurrent voice calls:
- Call initiation
- Audio streaming
- Status polling
- Call termination

### WebSocket Load Test (`websocket`)
Tests WebSocket connections:
- Connection establishment
- Message throughput
- Connection stability

### Concurrent Calls Test (`concurrent`)
Progressively increases concurrent calls to find capacity limits.

### Database Load Test (`database`)
Tests database performance:
- Simple queries
- Complex joins
- Write operations
- Aggregations

### Mixed Workload Test (`mixed`)
Combines all test types for realistic production simulation.

## Configuration

### Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--base-url` | `-u` | Target base URL |
| `--api-key` | `-k` | API key for authentication |
| `--org-id` | `-o` | Organization ID |
| `--test` | `-t` | Test type to run |
| `--users` | `-n` | Number of concurrent users |
| `--rps` | `-r` | Target requests per second |
| `--duration` | `-d` | Test duration in seconds |
| `--ramp-up` | | Ramp-up time in seconds |
| `--config` | `-c` | Config file path |
| `--profile` | `-p` | Profile name from config |
| `--output` | | Output directory |
| `--parallel` | | Run tests in parallel |
| `--quiet` | `-q` | Suppress progress output |

### Config File (YAML)

```yaml
target:
  base_url: "http://localhost:8000"
  api_key: "${BE_API_KEY}"
  organization_id: "${BE_ORG_ID}"

profiles:
  light:
    concurrent_users: 10
    requests_per_second: 50
    duration_seconds: 60

  heavy:
    concurrent_users: 100
    requests_per_second: 500
    duration_seconds: 600

thresholds:
  max_avg_response_time_ms: 500
  max_p95_response_time_ms: 1000
  max_error_rate: 0.01

output:
  directory: "./load_test_results"
  formats:
    - json
    - html
```

## Load Profiles

| Profile | Users | RPS | Duration | Use Case |
|---------|-------|-----|----------|----------|
| `smoke` | 5 | 10 | 30s | Quick validation |
| `light` | 10 | 50 | 60s | Development testing |
| `normal` | 50 | 200 | 5min | Standard load testing |
| `heavy` | 100 | 500 | 10min | Peak load simulation |
| `stress` | 200 | 1000 | 5min | Find breaking points |
| `soak` | 30 | 100 | 1hr | Stability testing |
| `spike` | 150 | 500 | 2min | Sudden load testing |

## Thresholds (SLOs)

Configure pass/fail criteria:

```yaml
thresholds:
  max_avg_response_time_ms: 500    # Average response time
  max_p95_response_time_ms: 1000   # 95th percentile
  max_p99_response_time_ms: 2000   # 99th percentile
  max_error_rate: 0.01             # 1% error rate
  min_requests_per_second: 50      # Minimum throughput
```

## Output Reports

### JSON Report
Detailed machine-readable results including:
- All request statistics
- Per-endpoint breakdowns
- Time series data
- Error analysis

### HTML Report
Interactive report with:
- Summary dashboard
- Charts and graphs
- Endpoint analysis
- Issue highlights
- Recommendations

### Markdown Report
CI/CD-friendly format for:
- Pull request comments
- Documentation
- Team notifications

## CI/CD Integration

### GitHub Actions

```yaml
jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[load-testing]"

      - name: Run load tests
        env:
          BE_API_KEY: ${{ secrets.BE_API_KEY }}
          BE_ORG_ID: ${{ secrets.BE_ORG_ID }}
        run: |
          python -m tests.load.cli \
            --test all \
            --profile light \
            --base-url ${{ vars.API_URL }}

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: load_test_results/
```

### GitLab CI

```yaml
load_test:
  stage: test
  script:
    - pip install -e ".[load-testing]"
    - python -m tests.load.cli --test all --profile light
  artifacts:
    paths:
      - load_test_results/
    reports:
      junit: load_test_results/*/summary.json
```

## Programmatic Usage

```python
import asyncio
from tests.load import (
    LoadTestConfig,
    LoadTestRunner,
    APILoadTest,
    CallSimulationTest,
    LoadTestReport,
    save_report,
)

async def run_custom_test():
    # Create configuration
    config = LoadTestConfig(
        base_url="http://localhost:8000",
        api_key="your-api-key",
        concurrent_users=50,
        requests_per_second=200,
        duration_seconds=300,
    )

    # Create runner and add tests
    runner = LoadTestRunner(config)
    runner.add_tests([APILoadTest, CallSimulationTest])

    # Run tests
    results = await runner.run_all()

    # Generate report
    report = LoadTestReport(results=results)
    report.analyze()

    # Save reports
    save_report(report, "./results", formats=["json", "html"])

    return report.overall_passed

# Run
asyncio.run(run_custom_test())
```

## Troubleshooting

### High Error Rates
- Check API key and organization ID
- Verify target URL is accessible
- Review server logs for errors
- Consider reducing load

### Timeout Errors
- Increase `request_timeout` setting
- Check network connectivity
- Review server performance

### Connection Errors
- Verify target is running
- Check firewall rules
- Increase connection pool size

### Memory Issues
- Reduce `save_detailed_results` for long tests
- Decrease concurrent users
- Run tests in smaller batches

## License

MIT License - See LICENSE file for details.

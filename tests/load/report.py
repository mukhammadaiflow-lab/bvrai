"""Load test report generation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .base import LoadTestResult


@dataclass
class LoadTestReport:
    """
    Load test report with analysis and recommendations.
    """
    title: str = "Load Test Report"
    generated_at: datetime = field(default_factory=datetime.utcnow)
    results: List[LoadTestResult] = field(default_factory=list)

    # Analysis
    overall_passed: bool = False
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Comparisons (if previous results available)
    comparison: Optional[Dict[str, Any]] = None

    def analyze(self) -> None:
        """Analyze results and generate insights."""
        self.critical_issues = []
        self.warnings = []
        self.recommendations = []

        passed_count = sum(1 for r in self.results if r.passed)
        self.overall_passed = passed_count == len(self.results)

        for result in self.results:
            self._analyze_result(result)

        self._generate_recommendations()

    def _analyze_result(self, result: LoadTestResult) -> None:
        """Analyze a single test result."""
        # Check for critical issues
        if result.error_rate > 0.05:
            self.critical_issues.append(
                f"{result.test_name}: High error rate ({result.error_rate:.1%})"
            )

        if result.p99_response_time_ms > 5000:
            self.critical_issues.append(
                f"{result.test_name}: Very high P99 latency ({result.p99_response_time_ms:.0f}ms)"
            )

        # Check for warnings
        if result.error_rate > 0.01:
            self.warnings.append(
                f"{result.test_name}: Elevated error rate ({result.error_rate:.1%})"
            )

        if result.p95_response_time_ms > 1000:
            self.warnings.append(
                f"{result.test_name}: High P95 latency ({result.p95_response_time_ms:.0f}ms)"
            )

        if result.std_dev_response_time_ms > result.avg_response_time_ms:
            self.warnings.append(
                f"{result.test_name}: High response time variance (std dev: {result.std_dev_response_time_ms:.0f}ms)"
            )

        # Check endpoint-specific issues
        for endpoint, stats in result.endpoint_stats.items():
            if stats.get("error_rate", 0) > 0.1:
                self.warnings.append(
                    f"{result.test_name}: Endpoint '{endpoint}' has high error rate ({stats['error_rate']:.1%})"
                )

    def _generate_recommendations(self) -> None:
        """Generate recommendations based on analysis."""
        # Response time recommendations
        high_latency_tests = [
            r for r in self.results
            if r.p95_response_time_ms > 500
        ]
        if high_latency_tests:
            self.recommendations.append(
                "Consider optimizing slow endpoints or adding caching to reduce response times"
            )

        # Error rate recommendations
        error_tests = [r for r in self.results if r.error_rate > 0.01]
        if error_tests:
            error_types = {}
            for r in error_tests:
                for error_type, count in r.errors_by_type.items():
                    error_types[error_type] = error_types.get(error_type, 0) + count

            if "timeout" in str(error_types).lower():
                self.recommendations.append(
                    "Increase request timeouts or optimize slow operations to reduce timeout errors"
                )

            if any("connection" in str(e).lower() for e in error_types):
                self.recommendations.append(
                    "Review connection pool settings and increase pool size if needed"
                )

        # Throughput recommendations
        low_throughput = [
            r for r in self.results
            if r.requests_per_second < r.config.min_requests_per_second
        ]
        if low_throughput:
            self.recommendations.append(
                "Consider horizontal scaling or optimizing bottlenecks to improve throughput"
            )

        # General recommendations
        if not self.overall_passed:
            self.recommendations.append(
                "Review failed test cases and address performance issues before production deployment"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "overall_passed": self.overall_passed,
            "summary": {
                "total_tests": len(self.results),
                "passed_tests": sum(1 for r in self.results if r.passed),
                "failed_tests": sum(1 for r in self.results if not r.passed),
                "total_requests": sum(r.total_requests for r in self.results),
                "total_errors": sum(r.failed_requests for r in self.results),
            },
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "results": [r.to_dict() for r in self.results],
            "comparison": self.comparison,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def generate_html_report(report: LoadTestReport) -> str:
    """Generate an HTML report."""
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-700: #374151;
            --gray-900: #111827;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--gray-900);
            background: var(--gray-100);
            padding: 2rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            background: white;
            padding: 2rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }}

        h1 {{
            font-size: 1.875rem;
            margin-bottom: 0.5rem;
        }}

        .timestamp {{
            color: var(--gray-700);
            font-size: 0.875rem;
        }}

        .status {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
            margin-left: 1rem;
        }}

        .status.passed {{
            background: #d1fae5;
            color: #065f46;
        }}

        .status.failed {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .card {{
            background: white;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        .card h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--gray-200);
        }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}

        .metric {{
            padding: 1rem;
            background: var(--gray-100);
            border-radius: 0.25rem;
        }}

        .metric-label {{
            font-size: 0.875rem;
            color: var(--gray-700);
        }}

        .metric-value {{
            font-size: 1.5rem;
            font-weight: 600;
        }}

        .issues-list {{
            list-style: none;
        }}

        .issues-list li {{
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-radius: 0.25rem;
        }}

        .issues-list li.critical {{
            background: #fee2e2;
            border-left: 4px solid var(--danger);
        }}

        .issues-list li.warning {{
            background: #fef3c7;
            border-left: 4px solid var(--warning);
        }}

        .issues-list li.recommendation {{
            background: #dbeafe;
            border-left: 4px solid var(--primary);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
        }}

        th {{
            background: var(--gray-100);
            font-weight: 600;
        }}

        tr:hover {{
            background: var(--gray-100);
        }}

        .badge {{
            display: inline-block;
            padding: 0.125rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .badge.success {{
            background: #d1fae5;
            color: #065f46;
        }}

        .badge.danger {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .chart-container {{
            height: 300px;
            margin-top: 1rem;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>
                {report.title}
                <span class="status {'passed' if report.overall_passed else 'failed'}">
                    {'PASSED' if report.overall_passed else 'FAILED'}
                </span>
            </h1>
            <p class="timestamp">Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </header>

        <div class="card">
            <h2>Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Total Tests</div>
                    <div class="metric-value">{len(report.results)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Passed</div>
                    <div class="metric-value" style="color: var(--success)">
                        {sum(1 for r in report.results if r.passed)}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Failed</div>
                    <div class="metric-value" style="color: var(--danger)">
                        {sum(1 for r in report.results if not r.passed)}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Requests</div>
                    <div class="metric-value">{sum(r.total_requests for r in report.results):,}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Errors</div>
                    <div class="metric-value">{sum(r.failed_requests for r in report.results):,}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Response Time</div>
                    <div class="metric-value">
                        {sum(r.avg_response_time_ms for r in report.results) / len(report.results) if report.results else 0:.0f}ms
                    </div>
                </div>
            </div>
        </div>

        {''.join(_generate_issues_html(report))}

        <div class="card">
            <h2>Test Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Status</th>
                        <th>Requests</th>
                        <th>Error Rate</th>
                        <th>Avg Response</th>
                        <th>P95 Response</th>
                        <th>P99 Response</th>
                        <th>Throughput</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(_generate_test_row(r) for r in report.results)}
                </tbody>
            </table>
        </div>

        {''.join(_generate_test_details(r) for r in report.results)}
    </div>

    <script>
        // Add charts if time series data is available
        {_generate_charts_js(report)}
    </script>
</body>
</html>
"""
    return html


def _generate_issues_html(report: LoadTestReport) -> List[str]:
    """Generate HTML for issues and recommendations."""
    sections = []

    if report.critical_issues:
        sections.append(f"""
        <div class="card">
            <h2>Critical Issues</h2>
            <ul class="issues-list">
                {''.join(f'<li class="critical">{issue}</li>' for issue in report.critical_issues)}
            </ul>
        </div>
        """)

    if report.warnings:
        sections.append(f"""
        <div class="card">
            <h2>Warnings</h2>
            <ul class="issues-list">
                {''.join(f'<li class="warning">{warning}</li>' for warning in report.warnings)}
            </ul>
        </div>
        """)

    if report.recommendations:
        sections.append(f"""
        <div class="card">
            <h2>Recommendations</h2>
            <ul class="issues-list">
                {''.join(f'<li class="recommendation">{rec}</li>' for rec in report.recommendations)}
            </ul>
        </div>
        """)

    return sections


def _generate_test_row(result: LoadTestResult) -> str:
    """Generate table row for a test result."""
    return f"""
    <tr>
        <td>{result.test_name}</td>
        <td>
            <span class="badge {'success' if result.passed else 'danger'}">
                {'PASSED' if result.passed else 'FAILED'}
            </span>
        </td>
        <td>{result.total_requests:,}</td>
        <td>{result.error_rate:.2%}</td>
        <td>{result.avg_response_time_ms:.0f}ms</td>
        <td>{result.p95_response_time_ms:.0f}ms</td>
        <td>{result.p99_response_time_ms:.0f}ms</td>
        <td>{result.requests_per_second:.1f} req/s</td>
    </tr>
    """


def _generate_test_details(result: LoadTestResult) -> str:
    """Generate detailed section for a test result."""
    endpoint_rows = ""
    for endpoint, stats in result.endpoint_stats.items():
        endpoint_rows += f"""
        <tr>
            <td>{endpoint}</td>
            <td>{stats.get('count', 0):,}</td>
            <td>{stats.get('success', 0):,}</td>
            <td>{stats.get('failed', 0):,}</td>
            <td>{stats.get('avg_response_time_ms', 0):.0f}ms</td>
            <td>{stats.get('p95_response_time_ms', 0):.0f}ms</td>
        </tr>
        """

    response_code_rows = ""
    for code, count in sorted(result.response_codes.items()):
        response_code_rows += f"<tr><td>{code}</td><td>{count:,}</td></tr>"

    failure_reasons = ""
    if result.failure_reasons:
        failure_reasons = f"""
        <h3>Failure Reasons</h3>
        <ul class="issues-list">
            {''.join(f'<li class="critical">{reason}</li>' for reason in result.failure_reasons)}
        </ul>
        """

    return f"""
    <div class="card">
        <h2>{result.test_name} - Details</h2>

        <div class="metrics" style="margin-bottom: 1.5rem;">
            <div class="metric">
                <div class="metric-label">Duration</div>
                <div class="metric-value">{result.duration_seconds:.1f}s</div>
            </div>
            <div class="metric">
                <div class="metric-label">Min Response</div>
                <div class="metric-value">{result.min_response_time_ms:.0f}ms</div>
            </div>
            <div class="metric">
                <div class="metric-label">Max Response</div>
                <div class="metric-value">{result.max_response_time_ms:.0f}ms</div>
            </div>
            <div class="metric">
                <div class="metric-label">Std Deviation</div>
                <div class="metric-value">{result.std_dev_response_time_ms:.0f}ms</div>
            </div>
        </div>

        {failure_reasons}

        <h3>Endpoint Statistics</h3>
        <table>
            <thead>
                <tr>
                    <th>Endpoint</th>
                    <th>Requests</th>
                    <th>Success</th>
                    <th>Failed</th>
                    <th>Avg Response</th>
                    <th>P95 Response</th>
                </tr>
            </thead>
            <tbody>
                {endpoint_rows if endpoint_rows else '<tr><td colspan="6">No endpoint data</td></tr>'}
            </tbody>
        </table>

        <h3 style="margin-top: 1.5rem;">Response Codes</h3>
        <table style="width: auto;">
            <thead>
                <tr>
                    <th>Code</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
                {response_code_rows if response_code_rows else '<tr><td colspan="2">No response codes</td></tr>'}
            </tbody>
        </table>

        <div id="chart-{result.test_name.replace(' ', '-')}" class="chart-container"></div>
    </div>
    """


def _generate_charts_js(report: LoadTestReport) -> str:
    """Generate JavaScript for charts."""
    charts_js = ""

    for result in report.results:
        if not result.time_series:
            continue

        chart_id = result.test_name.replace(' ', '-')
        labels = [f"{d['elapsed_seconds']:.0f}s" for d in result.time_series]
        response_times = [d.get('avg_response_time_ms', 0) for d in result.time_series]
        throughput = [d.get('requests_per_second', 0) for d in result.time_series]
        error_rates = [d.get('error_rate', 0) * 100 for d in result.time_series]

        charts_js += f"""
        new Chart(document.getElementById('chart-{chart_id}'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [
                    {{
                        label: 'Avg Response Time (ms)',
                        data: {json.dumps(response_times)},
                        borderColor: '#2563eb',
                        yAxisID: 'y',
                    }},
                    {{
                        label: 'Requests/sec',
                        data: {json.dumps(throughput)},
                        borderColor: '#10b981',
                        yAxisID: 'y1',
                    }},
                    {{
                        label: 'Error Rate (%)',
                        data: {json.dumps(error_rates)},
                        borderColor: '#ef4444',
                        yAxisID: 'y2',
                    }}
                ]
            }},
            options: {{
                responsive: true,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'Response Time (ms)'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Requests/sec'
                        }},
                        grid: {{
                            drawOnChartArea: false,
                        }},
                    }},
                    y2: {{
                        type: 'linear',
                        display: false,
                        min: 0,
                        max: 100,
                    }}
                }}
            }}
        }});
        """

    return charts_js


def save_report(
    report: LoadTestReport,
    output_dir: str,
    formats: List[str] = None,
) -> Dict[str, str]:
    """
    Save report in multiple formats.

    Args:
        report: The report to save
        output_dir: Directory to save reports
        formats: List of formats ('json', 'html', 'markdown')

    Returns:
        Dictionary mapping format to file path
    """
    formats = formats or ["json", "html"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    if "json" in formats:
        json_path = output_path / f"report_{timestamp}.json"
        with open(json_path, "w") as f:
            f.write(report.to_json())
        saved_files["json"] = str(json_path)

    if "html" in formats:
        html_path = output_path / f"report_{timestamp}.html"
        with open(html_path, "w") as f:
            f.write(generate_html_report(report))
        saved_files["html"] = str(html_path)

    if "markdown" in formats:
        md_path = output_path / f"report_{timestamp}.md"
        with open(md_path, "w") as f:
            f.write(_generate_markdown_report(report))
        saved_files["markdown"] = str(md_path)

    return saved_files


def _generate_markdown_report(report: LoadTestReport) -> str:
    """Generate Markdown report."""
    md = f"""# {report.title}

**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
**Status:** {'✅ PASSED' if report.overall_passed else '❌ FAILED'}

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {len(report.results)} |
| Passed | {sum(1 for r in report.results if r.passed)} |
| Failed | {sum(1 for r in report.results if not r.passed)} |
| Total Requests | {sum(r.total_requests for r in report.results):,} |
| Total Errors | {sum(r.failed_requests for r in report.results):,} |

"""

    if report.critical_issues:
        md += "## Critical Issues\n\n"
        for issue in report.critical_issues:
            md += f"- ❌ {issue}\n"
        md += "\n"

    if report.warnings:
        md += "## Warnings\n\n"
        for warning in report.warnings:
            md += f"- ⚠️ {warning}\n"
        md += "\n"

    if report.recommendations:
        md += "## Recommendations\n\n"
        for rec in report.recommendations:
            md += f"- 💡 {rec}\n"
        md += "\n"

    md += "## Test Results\n\n"
    md += "| Test | Status | Requests | Error Rate | Avg Response | P95 | Throughput |\n"
    md += "|------|--------|----------|------------|--------------|-----|------------|\n"

    for r in report.results:
        status = "✅" if r.passed else "❌"
        md += f"| {r.test_name} | {status} | {r.total_requests:,} | {r.error_rate:.2%} | {r.avg_response_time_ms:.0f}ms | {r.p95_response_time_ms:.0f}ms | {r.requests_per_second:.1f}/s |\n"

    return md

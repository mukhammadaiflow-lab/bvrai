"""
Metrics and Telemetry System
============================

Comprehensive observability infrastructure for the Voice AI platform including
metrics collection, distributed tracing, and telemetry export.

Author: Platform Observability Team
Version: 2.0.0
"""

from bvrai_core.telemetry.metrics import (
    MetricType,
    MetricValue,
    Metric,
    Counter,
    Gauge,
    Histogram,
    Timer,
    MetricsRegistry,
    get_registry,
    counter,
    gauge,
    histogram,
    timer,
)
from bvrai_core.telemetry.collectors import (
    MetricCollector,
    SystemMetricCollector,
    ProcessMetricCollector,
    ApplicationMetricCollector,
    VoiceMetricCollector,
    CollectorManager,
)
from bvrai_core.telemetry.exporters import (
    MetricExporter,
    PrometheusExporter,
    StatsDExporter,
    CloudWatchExporter,
    OpenTelemetryExporter,
    ExporterManager,
)
from bvrai_core.telemetry.tracing import (
    Span,
    SpanContext,
    SpanKind,
    SpanStatus,
    Tracer,
    TracerProvider,
    TraceExporter,
    JaegerExporter,
    ZipkinExporter,
    trace,
    get_current_span,
    inject_context,
    extract_context,
)

__all__ = [
    # Metrics
    "MetricType",
    "MetricValue",
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "MetricsRegistry",
    "get_registry",
    "counter",
    "gauge",
    "histogram",
    "timer",
    # Collectors
    "MetricCollector",
    "SystemMetricCollector",
    "ProcessMetricCollector",
    "ApplicationMetricCollector",
    "VoiceMetricCollector",
    "CollectorManager",
    # Exporters
    "MetricExporter",
    "PrometheusExporter",
    "StatsDExporter",
    "CloudWatchExporter",
    "OpenTelemetryExporter",
    "ExporterManager",
    # Tracing
    "Span",
    "SpanContext",
    "SpanKind",
    "SpanStatus",
    "Tracer",
    "TracerProvider",
    "TraceExporter",
    "JaegerExporter",
    "ZipkinExporter",
    "trace",
    "get_current_span",
    "inject_context",
    "extract_context",
]

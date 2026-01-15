"""
Metric Exporters
================

Export metrics to various backends including Prometheus, StatsD, CloudWatch,
and OpenTelemetry.

Author: Platform Observability Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

import structlog

from platform.telemetry.metrics import (
    Counter,
    Gauge,
    Histogram,
    Metric,
    MetricType,
    MetricValue,
    MetricsRegistry,
    Timer,
    get_registry,
)

logger = structlog.get_logger(__name__)


@dataclass
class ExporterConfig:
    """Base configuration for exporters"""

    enabled: bool = True
    interval_seconds: float = 15.0
    timeout_seconds: float = 10.0
    batch_size: int = 100
    prefix: str = ""
    default_labels: Dict[str, str] = field(default_factory=dict)


class MetricExporter(ABC):
    """
    Base class for metric exporters.

    Exporters push metrics to external monitoring systems.
    """

    def __init__(
        self,
        config: ExporterConfig,
        registry: Optional[MetricsRegistry] = None,
    ):
        self.config = config
        self._registry = registry or get_registry()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._logger = structlog.get_logger(self.__class__.__name__)
        self._export_count = 0
        self._error_count = 0
        self._last_export: Optional[datetime] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Exporter name"""
        pass

    @abstractmethod
    async def export(self, metrics: Dict[str, List[MetricValue]]) -> bool:
        """Export metrics"""
        pass

    async def start(self) -> None:
        """Start the exporter"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._export_loop())
        self._logger.info("exporter_started", name=self.name)

    async def stop(self) -> None:
        """Stop the exporter"""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._logger.info("exporter_stopped", name=self.name)

    async def _export_loop(self) -> None:
        """Main export loop"""
        while self._running:
            try:
                metrics = self._registry.collect_all()
                success = await self.export(metrics)

                if success:
                    self._export_count += 1
                    self._last_export = datetime.utcnow()
                else:
                    self._error_count += 1
            except Exception as e:
                self._error_count += 1
                self._logger.error(
                    "export_error",
                    exporter=self.name,
                    error=str(e),
                )

            await asyncio.sleep(self.config.interval_seconds)

    def _with_labels(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Merge with default labels"""
        return {**self.config.default_labels, **labels}


class PrometheusExporter(MetricExporter):
    """
    Exports metrics in Prometheus format.

    Can either push to Pushgateway or serve metrics via HTTP endpoint.
    """

    @dataclass
    class Config(ExporterConfig):
        """Prometheus exporter configuration"""

        pushgateway_url: Optional[str] = None
        job_name: str = "voice_ai"
        instance: Optional[str] = None
        honor_labels: bool = True

    def __init__(
        self,
        config: Optional[Config] = None,
        registry: Optional[MetricsRegistry] = None,
    ):
        super().__init__(config or self.Config(), registry)
        self._config: PrometheusExporter.Config = self.config  # type: ignore

    @property
    def name(self) -> str:
        return "prometheus"

    async def export(self, metrics: Dict[str, List[MetricValue]]) -> bool:
        """Export metrics to Prometheus"""
        if not self._config.pushgateway_url:
            # Just format - assume scrape-based collection
            return True

        try:
            output = self._format_metrics(metrics)

            # Push to gateway
            url = urljoin(
                self._config.pushgateway_url,
                f"/metrics/job/{self._config.job_name}",
            )
            if self._config.instance:
                url += f"/instance/{self._config.instance}"

            # Use aiohttp for async HTTP (simplified for this implementation)
            self._logger.debug("prometheus_push", url=url, metrics_count=len(metrics))
            return True
        except Exception as e:
            self._logger.error("prometheus_export_error", error=str(e))
            return False

    def _format_metrics(self, metrics: Dict[str, List[MetricValue]]) -> str:
        """Format metrics in Prometheus exposition format"""
        lines = []

        for name, values in metrics.items():
            if not values:
                continue

            # Add TYPE and HELP comments
            metric = self._registry.get(name)
            if metric:
                lines.append(f"# HELP {name} {metric.description}")
                prom_type = self._prometheus_type(metric.type)
                lines.append(f"# TYPE {name} {prom_type}")

            for value in values:
                labels = self._with_labels(value.labels)
                label_str = self._format_labels(labels)
                lines.append(f"{name}{label_str} {value.value}")

        return "\n".join(lines) + "\n"

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus"""
        if not labels:
            return ""

        pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(pairs) + "}"

    def _prometheus_type(self, metric_type: MetricType) -> str:
        """Convert to Prometheus type"""
        mapping = {
            MetricType.COUNTER: "counter",
            MetricType.GAUGE: "gauge",
            MetricType.HISTOGRAM: "histogram",
            MetricType.TIMER: "histogram",
            MetricType.SUMMARY: "summary",
        }
        return mapping.get(metric_type, "gauge")

    def render(self) -> str:
        """Render metrics for HTTP endpoint"""
        metrics = self._registry.collect_all()
        return self._format_metrics(metrics)


class StatsDExporter(MetricExporter):
    """
    Exports metrics to StatsD.

    Supports standard StatsD protocol and DogStatsD extensions.
    """

    @dataclass
    class Config(ExporterConfig):
        """StatsD exporter configuration"""

        host: str = "localhost"
        port: int = 8125
        protocol: str = "udp"  # udp or tcp
        use_dogstatsd: bool = False
        max_buffer_size: int = 512

    def __init__(
        self,
        config: Optional[Config] = None,
        registry: Optional[MetricsRegistry] = None,
    ):
        super().__init__(config or self.Config(), registry)
        self._config: StatsDExporter.Config = self.config  # type: ignore
        self._socket: Optional[socket.socket] = None

    @property
    def name(self) -> str:
        return "statsd"

    async def start(self) -> None:
        """Start the exporter"""
        self._connect()
        await super().start()

    async def stop(self) -> None:
        """Stop the exporter"""
        await super().stop()
        self._disconnect()

    def _connect(self) -> None:
        """Connect to StatsD"""
        if self._config.protocol == "udp":
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self._config.host, self._config.port))

        self._socket.setblocking(False)

    def _disconnect(self) -> None:
        """Disconnect from StatsD"""
        if self._socket:
            self._socket.close()
            self._socket = None

    async def export(self, metrics: Dict[str, List[MetricValue]]) -> bool:
        """Export metrics to StatsD"""
        if not self._socket:
            self._connect()

        try:
            messages = self._format_metrics(metrics)

            for message in messages:
                await self._send(message)

            return True
        except Exception as e:
            self._logger.error("statsd_export_error", error=str(e))
            return False

    async def _send(self, message: str) -> None:
        """Send message to StatsD"""
        if not self._socket:
            return

        data = message.encode("utf-8")

        loop = asyncio.get_event_loop()
        if self._config.protocol == "udp":
            await loop.run_in_executor(
                None,
                lambda: self._socket.sendto(
                    data,
                    (self._config.host, self._config.port),
                ),
            )
        else:
            await loop.run_in_executor(
                None,
                lambda: self._socket.sendall(data),
            )

    def _format_metrics(self, metrics: Dict[str, List[MetricValue]]) -> List[str]:
        """Format metrics for StatsD"""
        messages = []

        for name, values in metrics.items():
            metric = self._registry.get(name)
            if not metric:
                continue

            for value in values:
                msg = self._format_metric(name, value, metric.type)
                if msg:
                    messages.append(msg)

        return messages

    def _format_metric(
        self,
        name: str,
        value: MetricValue,
        metric_type: MetricType,
    ) -> str:
        """Format a single metric"""
        prefix = self.config.prefix
        full_name = f"{prefix}.{name}" if prefix else name

        # Add labels as tags for DogStatsD
        tags = ""
        if self._config.use_dogstatsd and value.labels:
            tag_list = [f"{k}:{v}" for k, v in value.labels.items()]
            tags = "|#" + ",".join(tag_list)

        statsd_type = self._statsd_type(metric_type)

        return f"{full_name}:{value.value}|{statsd_type}{tags}"

    def _statsd_type(self, metric_type: MetricType) -> str:
        """Convert to StatsD type"""
        mapping = {
            MetricType.COUNTER: "c",
            MetricType.GAUGE: "g",
            MetricType.HISTOGRAM: "h",
            MetricType.TIMER: "ms",
            MetricType.SUMMARY: "h",
        }
        return mapping.get(metric_type, "g")


class CloudWatchExporter(MetricExporter):
    """
    Exports metrics to AWS CloudWatch.
    """

    @dataclass
    class Config(ExporterConfig):
        """CloudWatch exporter configuration"""

        namespace: str = "VoiceAI"
        region: str = "us-east-1"
        dimensions: Dict[str, str] = field(default_factory=dict)
        storage_resolution: int = 60  # 1 or 60 seconds
        access_key_id: Optional[str] = None
        secret_access_key: Optional[str] = None

    def __init__(
        self,
        config: Optional[Config] = None,
        registry: Optional[MetricsRegistry] = None,
    ):
        super().__init__(config or self.Config(), registry)
        self._config: CloudWatchExporter.Config = self.config  # type: ignore

    @property
    def name(self) -> str:
        return "cloudwatch"

    async def export(self, metrics: Dict[str, List[MetricValue]]) -> bool:
        """Export metrics to CloudWatch"""
        try:
            metric_data = self._format_metrics(metrics)

            if not metric_data:
                return True

            # Batch metrics (CloudWatch allows 150 metrics per request)
            for i in range(0, len(metric_data), 150):
                batch = metric_data[i : i + 150]
                await self._put_metrics(batch)

            return True
        except Exception as e:
            self._logger.error("cloudwatch_export_error", error=str(e))
            return False

    async def _put_metrics(self, metric_data: List[Dict[str, Any]]) -> None:
        """Put metrics to CloudWatch"""
        # In a real implementation, this would use boto3
        self._logger.debug(
            "cloudwatch_put",
            namespace=self._config.namespace,
            metrics_count=len(metric_data),
        )

    def _format_metrics(self, metrics: Dict[str, List[MetricValue]]) -> List[Dict[str, Any]]:
        """Format metrics for CloudWatch"""
        metric_data = []

        for name, values in metrics.items():
            metric = self._registry.get(name)
            if not metric:
                continue

            for value in values:
                data = self._format_metric_data(name, value, metric.type)
                if data:
                    metric_data.append(data)

        return metric_data

    def _format_metric_data(
        self,
        name: str,
        value: MetricValue,
        metric_type: MetricType,
    ) -> Optional[Dict[str, Any]]:
        """Format a single metric for CloudWatch"""
        dimensions = [
            {"Name": k, "Value": v}
            for k, v in {**self._config.dimensions, **value.labels}.items()
        ]

        unit = self._cloudwatch_unit(metric_type)

        return {
            "MetricName": name,
            "Dimensions": dimensions,
            "Timestamp": value.timestamp.isoformat(),
            "Value": value.value,
            "Unit": unit,
            "StorageResolution": self._config.storage_resolution,
        }

    def _cloudwatch_unit(self, metric_type: MetricType) -> str:
        """Get CloudWatch unit"""
        mapping = {
            MetricType.COUNTER: "Count",
            MetricType.GAUGE: "None",
            MetricType.HISTOGRAM: "None",
            MetricType.TIMER: "Milliseconds",
            MetricType.SUMMARY: "None",
        }
        return mapping.get(metric_type, "None")


class OpenTelemetryExporter(MetricExporter):
    """
    Exports metrics using OpenTelemetry Protocol (OTLP).
    """

    @dataclass
    class Config(ExporterConfig):
        """OpenTelemetry exporter configuration"""

        endpoint: str = "http://localhost:4317"
        protocol: str = "grpc"  # grpc or http
        headers: Dict[str, str] = field(default_factory=dict)
        compression: str = "gzip"
        resource_attributes: Dict[str, str] = field(default_factory=dict)

    def __init__(
        self,
        config: Optional[Config] = None,
        registry: Optional[MetricsRegistry] = None,
    ):
        super().__init__(config or self.Config(), registry)
        self._config: OpenTelemetryExporter.Config = self.config  # type: ignore

    @property
    def name(self) -> str:
        return "opentelemetry"

    async def export(self, metrics: Dict[str, List[MetricValue]]) -> bool:
        """Export metrics via OTLP"""
        try:
            resource = self._build_resource()
            scope_metrics = self._format_metrics(metrics)

            payload = {
                "resourceMetrics": [
                    {
                        "resource": resource,
                        "scopeMetrics": [
                            {
                                "scope": {
                                    "name": "voice_ai.telemetry",
                                    "version": "2.0.0",
                                },
                                "metrics": scope_metrics,
                            }
                        ],
                    }
                ]
            }

            await self._send_otlp(payload)
            return True
        except Exception as e:
            self._logger.error("otel_export_error", error=str(e))
            return False

    async def _send_otlp(self, payload: Dict[str, Any]) -> None:
        """Send OTLP payload"""
        # In a real implementation, this would use gRPC or HTTP
        self._logger.debug(
            "otel_send",
            endpoint=self._config.endpoint,
            protocol=self._config.protocol,
        )

    def _build_resource(self) -> Dict[str, Any]:
        """Build OTLP resource"""
        attributes = [
            {"key": k, "value": {"stringValue": v}}
            for k, v in self._config.resource_attributes.items()
        ]

        return {"attributes": attributes}

    def _format_metrics(self, metrics: Dict[str, List[MetricValue]]) -> List[Dict[str, Any]]:
        """Format metrics for OTLP"""
        otlp_metrics = []

        for name, values in metrics.items():
            metric = self._registry.get(name)
            if not metric:
                continue

            otlp_metric = self._format_otlp_metric(name, values, metric)
            if otlp_metric:
                otlp_metrics.append(otlp_metric)

        return otlp_metrics

    def _format_otlp_metric(
        self,
        name: str,
        values: List[MetricValue],
        metric: Metric,
    ) -> Optional[Dict[str, Any]]:
        """Format a single metric for OTLP"""
        if not values:
            return None

        data_points = []
        for value in values:
            attributes = [
                {"key": k, "value": {"stringValue": v}}
                for k, v in value.labels.items()
            ]

            data_points.append({
                "attributes": attributes,
                "startTimeUnixNano": int(value.timestamp.timestamp() * 1e9),
                "timeUnixNano": int(time.time() * 1e9),
                "asDouble": value.value,
            })

        otlp_metric = {
            "name": name,
            "description": metric.description,
            "unit": metric.unit,
        }

        # Set metric type-specific data
        if metric.type == MetricType.COUNTER:
            otlp_metric["sum"] = {
                "dataPoints": data_points,
                "aggregationTemporality": 2,  # CUMULATIVE
                "isMonotonic": True,
            }
        elif metric.type == MetricType.GAUGE:
            otlp_metric["gauge"] = {
                "dataPoints": data_points,
            }
        elif metric.type in (MetricType.HISTOGRAM, MetricType.TIMER):
            otlp_metric["histogram"] = {
                "dataPoints": data_points,
                "aggregationTemporality": 2,
            }
        else:
            otlp_metric["gauge"] = {
                "dataPoints": data_points,
            }

        return otlp_metric


class ExporterManager:
    """
    Manages multiple metric exporters.

    Usage:
        manager = ExporterManager()
        manager.add(PrometheusExporter())
        manager.add(StatsDExporter())
        await manager.start_all()
    """

    def __init__(self):
        self._exporters: Dict[str, MetricExporter] = {}
        self._logger = structlog.get_logger("exporter_manager")

    def add(self, exporter: MetricExporter) -> None:
        """Add an exporter"""
        if exporter.name in self._exporters:
            raise ValueError(f"Exporter {exporter.name} already registered")
        self._exporters[exporter.name] = exporter
        self._logger.info("exporter_added", name=exporter.name)

    def remove(self, name: str) -> Optional[MetricExporter]:
        """Remove an exporter"""
        return self._exporters.pop(name, None)

    def get(self, name: str) -> Optional[MetricExporter]:
        """Get an exporter by name"""
        return self._exporters.get(name)

    async def start_all(self) -> None:
        """Start all exporters"""
        for exporter in self._exporters.values():
            if exporter.config.enabled:
                await exporter.start()

    async def stop_all(self) -> None:
        """Stop all exporters"""
        for exporter in self._exporters.values():
            await exporter.stop()

    def list_exporters(self) -> List[str]:
        """List all exporter names"""
        return list(self._exporters.keys())

    async def export_now(self) -> Dict[str, bool]:
        """Trigger immediate export from all exporters"""
        results = {}
        metrics = get_registry().collect_all()

        for name, exporter in self._exporters.items():
            if exporter.config.enabled:
                try:
                    results[name] = await exporter.export(metrics)
                except Exception as e:
                    self._logger.error(
                        "immediate_export_error",
                        exporter=name,
                        error=str(e),
                    )
                    results[name] = False

        return results

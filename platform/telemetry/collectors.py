"""
Metric Collectors
=================

Automated metric collection for system, process, and application metrics.

Author: Platform Observability Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import gc
import os
import platform
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from platform.telemetry.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    Timer,
    get_registry,
)

logger = structlog.get_logger(__name__)


@dataclass
class CollectorConfig:
    """Configuration for metric collectors"""

    enabled: bool = True
    interval_seconds: float = 15.0
    prefix: str = ""
    labels: Dict[str, str] = field(default_factory=dict)


class MetricCollector(ABC):
    """
    Base class for metric collectors.

    Collectors automatically gather metrics on a schedule.
    """

    def __init__(
        self,
        config: CollectorConfig,
        registry: Optional[MetricsRegistry] = None,
    ):
        self.config = config
        self._registry = registry or get_registry()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._logger = structlog.get_logger(self.__class__.__name__)
        self._last_collection: Optional[datetime] = None
        self._collection_count = 0
        self._error_count = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Collector name"""
        pass

    @abstractmethod
    async def collect(self) -> None:
        """Collect metrics"""
        pass

    async def start(self) -> None:
        """Start the collector"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._collection_loop())
        self._logger.info("collector_started", name=self.name)

    async def stop(self) -> None:
        """Stop the collector"""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._logger.info("collector_stopped", name=self.name)

    async def _collection_loop(self) -> None:
        """Main collection loop"""
        while self._running:
            try:
                await self.collect()
                self._last_collection = datetime.utcnow()
                self._collection_count += 1
            except Exception as e:
                self._error_count += 1
                self._logger.error(
                    "collection_error",
                    collector=self.name,
                    error=str(e),
                )

            await asyncio.sleep(self.config.interval_seconds)

    def _prefixed_name(self, name: str) -> str:
        """Get prefixed metric name"""
        if self.config.prefix:
            return f"{self.config.prefix}_{name}"
        return name

    def _with_labels(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Merge with default labels"""
        return {**self.config.labels, **labels}


class SystemMetricCollector(MetricCollector):
    """
    Collects system-level metrics.

    Metrics collected:
    - CPU usage (user, system, idle)
    - Memory usage (total, available, used, percent)
    - Disk usage (total, used, free)
    - Network I/O (bytes sent/received)
    - Load average
    """

    def __init__(
        self,
        config: Optional[CollectorConfig] = None,
        registry: Optional[MetricsRegistry] = None,
    ):
        super().__init__(config or CollectorConfig(prefix="system"), registry)
        self._setup_metrics()
        self._prev_cpu_times: Optional[Dict[str, float]] = None
        self._prev_net_io: Optional[Dict[str, int]] = None

    @property
    def name(self) -> str:
        return "system"

    def _setup_metrics(self) -> None:
        """Setup system metrics"""
        prefix = self._prefixed_name

        # CPU metrics
        self._cpu_usage = self._registry.gauge(
            prefix("cpu_usage_percent"),
            "CPU usage percentage",
            labels=["cpu", "mode"],
        )

        # Memory metrics
        self._memory_bytes = self._registry.gauge(
            prefix("memory_bytes"),
            "Memory in bytes",
            labels=["type"],
        )
        self._memory_percent = self._registry.gauge(
            prefix("memory_percent"),
            "Memory usage percentage",
        )

        # Disk metrics
        self._disk_bytes = self._registry.gauge(
            prefix("disk_bytes"),
            "Disk space in bytes",
            labels=["path", "type"],
        )
        self._disk_percent = self._registry.gauge(
            prefix("disk_percent"),
            "Disk usage percentage",
            labels=["path"],
        )

        # Network metrics
        self._network_bytes = self._registry.counter(
            prefix("network_bytes_total"),
            "Network bytes transferred",
            labels=["interface", "direction"],
        )

        # Load average
        self._load_average = self._registry.gauge(
            prefix("load_average"),
            "System load average",
            labels=["period"],
        )

    async def collect(self) -> None:
        """Collect system metrics"""
        await asyncio.gather(
            self._collect_cpu(),
            self._collect_memory(),
            self._collect_disk(),
            self._collect_network(),
            self._collect_load(),
        )

    async def _collect_cpu(self) -> None:
        """Collect CPU metrics"""
        try:
            # Read /proc/stat for CPU times
            with open("/proc/stat", "r") as f:
                for line in f:
                    if line.startswith("cpu "):
                        parts = line.split()
                        user = int(parts[1])
                        nice = int(parts[2])
                        system = int(parts[3])
                        idle = int(parts[4])
                        iowait = int(parts[5]) if len(parts) > 5 else 0

                        total = user + nice + system + idle + iowait

                        if self._prev_cpu_times:
                            prev_total = sum(self._prev_cpu_times.values())
                            diff_total = total - prev_total

                            if diff_total > 0:
                                user_pct = (user - self._prev_cpu_times.get("user", 0)) / diff_total * 100
                                system_pct = (system - self._prev_cpu_times.get("system", 0)) / diff_total * 100
                                idle_pct = (idle - self._prev_cpu_times.get("idle", 0)) / diff_total * 100

                                self._cpu_usage.set(
                                    user_pct,
                                    labels=self._with_labels({"cpu": "total", "mode": "user"}),
                                )
                                self._cpu_usage.set(
                                    system_pct,
                                    labels=self._with_labels({"cpu": "total", "mode": "system"}),
                                )
                                self._cpu_usage.set(
                                    idle_pct,
                                    labels=self._with_labels({"cpu": "total", "mode": "idle"}),
                                )

                        self._prev_cpu_times = {
                            "user": user,
                            "nice": nice,
                            "system": system,
                            "idle": idle,
                            "iowait": iowait,
                        }
                        break
        except Exception as e:
            self._logger.debug("cpu_collection_error", error=str(e))

    async def _collect_memory(self) -> None:
        """Collect memory metrics"""
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().split()[0]
                        meminfo[key] = int(value) * 1024  # Convert KB to bytes

            total = meminfo.get("MemTotal", 0)
            available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            used = total - available

            self._memory_bytes.set(
                total,
                labels=self._with_labels({"type": "total"}),
            )
            self._memory_bytes.set(
                available,
                labels=self._with_labels({"type": "available"}),
            )
            self._memory_bytes.set(
                used,
                labels=self._with_labels({"type": "used"}),
            )

            if total > 0:
                self._memory_percent.set(
                    (used / total) * 100,
                    labels=self._with_labels({}),
                )
        except Exception as e:
            self._logger.debug("memory_collection_error", error=str(e))

    async def _collect_disk(self) -> None:
        """Collect disk metrics"""
        try:
            statvfs = os.statvfs("/")
            total = statvfs.f_blocks * statvfs.f_frsize
            free = statvfs.f_bfree * statvfs.f_frsize
            used = total - free

            self._disk_bytes.set(
                total,
                labels=self._with_labels({"path": "/", "type": "total"}),
            )
            self._disk_bytes.set(
                used,
                labels=self._with_labels({"path": "/", "type": "used"}),
            )
            self._disk_bytes.set(
                free,
                labels=self._with_labels({"path": "/", "type": "free"}),
            )

            if total > 0:
                self._disk_percent.set(
                    (used / total) * 100,
                    labels=self._with_labels({"path": "/"}),
                )
        except Exception as e:
            self._logger.debug("disk_collection_error", error=str(e))

    async def _collect_network(self) -> None:
        """Collect network metrics"""
        try:
            with open("/proc/net/dev", "r") as f:
                lines = f.readlines()[2:]  # Skip headers

                for line in lines:
                    parts = line.split(":")
                    if len(parts) != 2:
                        continue

                    interface = parts[0].strip()
                    if interface == "lo":
                        continue

                    values = parts[1].split()
                    rx_bytes = int(values[0])
                    tx_bytes = int(values[8])

                    if self._prev_net_io:
                        prev_rx = self._prev_net_io.get(f"{interface}_rx", 0)
                        prev_tx = self._prev_net_io.get(f"{interface}_tx", 0)

                        if rx_bytes >= prev_rx:
                            self._network_bytes.inc(
                                rx_bytes - prev_rx,
                                labels=self._with_labels(
                                    {"interface": interface, "direction": "received"}
                                ),
                            )
                        if tx_bytes >= prev_tx:
                            self._network_bytes.inc(
                                tx_bytes - prev_tx,
                                labels=self._with_labels(
                                    {"interface": interface, "direction": "sent"}
                                ),
                            )

                    if self._prev_net_io is None:
                        self._prev_net_io = {}
                    self._prev_net_io[f"{interface}_rx"] = rx_bytes
                    self._prev_net_io[f"{interface}_tx"] = tx_bytes
        except Exception as e:
            self._logger.debug("network_collection_error", error=str(e))

    async def _collect_load(self) -> None:
        """Collect load average"""
        try:
            with open("/proc/loadavg", "r") as f:
                parts = f.read().split()
                load_1 = float(parts[0])
                load_5 = float(parts[1])
                load_15 = float(parts[2])

                self._load_average.set(
                    load_1,
                    labels=self._with_labels({"period": "1m"}),
                )
                self._load_average.set(
                    load_5,
                    labels=self._with_labels({"period": "5m"}),
                )
                self._load_average.set(
                    load_15,
                    labels=self._with_labels({"period": "15m"}),
                )
        except Exception as e:
            self._logger.debug("load_collection_error", error=str(e))


class ProcessMetricCollector(MetricCollector):
    """
    Collects process-level metrics.

    Metrics collected:
    - Process CPU time
    - Process memory (RSS, VMS)
    - Open file descriptors
    - Thread count
    - Python GC statistics
    """

    def __init__(
        self,
        config: Optional[CollectorConfig] = None,
        registry: Optional[MetricsRegistry] = None,
    ):
        super().__init__(config or CollectorConfig(prefix="process"), registry)
        self._setup_metrics()
        self._prev_cpu_times: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return "process"

    def _setup_metrics(self) -> None:
        """Setup process metrics"""
        prefix = self._prefixed_name

        # CPU
        self._cpu_seconds = self._registry.counter(
            prefix("cpu_seconds_total"),
            "Total CPU time in seconds",
            labels=["mode"],
        )

        # Memory
        self._memory_bytes = self._registry.gauge(
            prefix("memory_bytes"),
            "Process memory in bytes",
            labels=["type"],
        )

        # File descriptors
        self._open_fds = self._registry.gauge(
            prefix("open_fds"),
            "Number of open file descriptors",
        )
        self._max_fds = self._registry.gauge(
            prefix("max_fds"),
            "Maximum number of file descriptors",
        )

        # Threads
        self._thread_count = self._registry.gauge(
            prefix("threads"),
            "Number of threads",
        )

        # GC
        self._gc_collections = self._registry.counter(
            prefix("gc_collections_total"),
            "GC collections",
            labels=["generation"],
        )
        self._gc_objects = self._registry.gauge(
            prefix("gc_objects"),
            "GC tracked objects",
        )

        # Uptime
        self._start_time = self._registry.gauge(
            prefix("start_time_seconds"),
            "Process start time in Unix seconds",
        )

        # Python info
        self._python_info = self._registry.gauge(
            prefix("python_info"),
            "Python version info",
            labels=["version", "implementation"],
        )

    async def collect(self) -> None:
        """Collect process metrics"""
        await asyncio.gather(
            self._collect_cpu(),
            self._collect_memory(),
            self._collect_fds(),
            self._collect_threads(),
            self._collect_gc(),
            self._collect_info(),
        )

    async def _collect_cpu(self) -> None:
        """Collect CPU metrics"""
        try:
            with open(f"/proc/{os.getpid()}/stat", "r") as f:
                parts = f.read().split()
                utime = int(parts[13]) / os.sysconf("SC_CLK_TCK")
                stime = int(parts[14]) / os.sysconf("SC_CLK_TCK")

                if self._prev_cpu_times:
                    user_diff = utime - self._prev_cpu_times.get("user", 0)
                    system_diff = stime - self._prev_cpu_times.get("system", 0)

                    if user_diff > 0:
                        self._cpu_seconds.inc(
                            user_diff,
                            labels=self._with_labels({"mode": "user"}),
                        )
                    if system_diff > 0:
                        self._cpu_seconds.inc(
                            system_diff,
                            labels=self._with_labels({"mode": "system"}),
                        )

                self._prev_cpu_times = {"user": utime, "system": stime}
        except Exception as e:
            self._logger.debug("cpu_collection_error", error=str(e))

    async def _collect_memory(self) -> None:
        """Collect memory metrics"""
        try:
            with open(f"/proc/{os.getpid()}/statm", "r") as f:
                parts = f.read().split()
                page_size = os.sysconf("SC_PAGE_SIZE")
                vms = int(parts[0]) * page_size
                rss = int(parts[1]) * page_size

                self._memory_bytes.set(
                    rss,
                    labels=self._with_labels({"type": "rss"}),
                )
                self._memory_bytes.set(
                    vms,
                    labels=self._with_labels({"type": "vms"}),
                )
        except Exception as e:
            self._logger.debug("memory_collection_error", error=str(e))

    async def _collect_fds(self) -> None:
        """Collect file descriptor metrics"""
        try:
            fd_path = f"/proc/{os.getpid()}/fd"
            open_fds = len(os.listdir(fd_path))
            self._open_fds.set(open_fds, labels=self._with_labels({}))

            with open(f"/proc/{os.getpid()}/limits", "r") as f:
                for line in f:
                    if "Max open files" in line:
                        parts = line.split()
                        max_fds = int(parts[3])
                        self._max_fds.set(max_fds, labels=self._with_labels({}))
                        break
        except Exception as e:
            self._logger.debug("fd_collection_error", error=str(e))

    async def _collect_threads(self) -> None:
        """Collect thread metrics"""
        try:
            self._thread_count.set(
                threading.active_count(),
                labels=self._with_labels({}),
            )
        except Exception as e:
            self._logger.debug("thread_collection_error", error=str(e))

    async def _collect_gc(self) -> None:
        """Collect garbage collection metrics"""
        try:
            stats = gc.get_stats()
            for i, gen_stats in enumerate(stats):
                self._gc_collections.inc(
                    0,  # We just want to track the gauge
                    labels=self._with_labels({"generation": str(i)}),
                )

            self._gc_objects.set(
                len(gc.get_objects()),
                labels=self._with_labels({}),
            )
        except Exception as e:
            self._logger.debug("gc_collection_error", error=str(e))

    async def _collect_info(self) -> None:
        """Collect process info"""
        try:
            with open(f"/proc/{os.getpid()}/stat", "r") as f:
                parts = f.read().split()
                start_time_ticks = int(parts[21])

            with open("/proc/stat", "r") as f:
                for line in f:
                    if line.startswith("btime"):
                        boot_time = int(line.split()[1])
                        break

            start_time = boot_time + (start_time_ticks / os.sysconf("SC_CLK_TCK"))
            self._start_time.set(start_time, labels=self._with_labels({}))

            self._python_info.set(
                1,
                labels=self._with_labels({
                    "version": platform.python_version(),
                    "implementation": platform.python_implementation(),
                }),
            )
        except Exception as e:
            self._logger.debug("info_collection_error", error=str(e))


class ApplicationMetricCollector(MetricCollector):
    """
    Collects application-level metrics.

    Provides metrics for:
    - HTTP request latency and throughput
    - Database query performance
    - Cache hit rates
    - Queue depths
    - Error rates
    """

    def __init__(
        self,
        config: Optional[CollectorConfig] = None,
        registry: Optional[MetricsRegistry] = None,
    ):
        super().__init__(config or CollectorConfig(prefix="app"), registry)
        self._setup_metrics()
        self._custom_gauges: Dict[str, Callable[[], float]] = {}

    @property
    def name(self) -> str:
        return "application"

    def _setup_metrics(self) -> None:
        """Setup application metrics"""
        prefix = self._prefixed_name

        # HTTP
        self.http_requests = self._registry.counter(
            prefix("http_requests_total"),
            "Total HTTP requests",
            labels=["method", "endpoint", "status"],
        )
        self.http_latency = self._registry.histogram(
            prefix("http_request_duration_seconds"),
            "HTTP request latency",
            labels=["method", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )
        self.http_in_progress = self._registry.gauge(
            prefix("http_requests_in_progress"),
            "HTTP requests in progress",
            labels=["method", "endpoint"],
        )

        # Database
        self.db_queries = self._registry.counter(
            prefix("db_queries_total"),
            "Total database queries",
            labels=["operation", "table"],
        )
        self.db_latency = self._registry.histogram(
            prefix("db_query_duration_seconds"),
            "Database query latency",
            labels=["operation"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )
        self.db_connections = self._registry.gauge(
            prefix("db_connections"),
            "Database connections",
            labels=["state"],
        )

        # Cache
        self.cache_hits = self._registry.counter(
            prefix("cache_hits_total"),
            "Cache hits",
            labels=["cache"],
        )
        self.cache_misses = self._registry.counter(
            prefix("cache_misses_total"),
            "Cache misses",
            labels=["cache"],
        )
        self.cache_size = self._registry.gauge(
            prefix("cache_size"),
            "Cache size in items",
            labels=["cache"],
        )

        # Queue
        self.queue_depth = self._registry.gauge(
            prefix("queue_depth"),
            "Queue depth",
            labels=["queue"],
        )
        self.queue_latency = self._registry.histogram(
            prefix("queue_wait_seconds"),
            "Queue wait time",
            labels=["queue"],
        )

        # Errors
        self.errors = self._registry.counter(
            prefix("errors_total"),
            "Total errors",
            labels=["type", "component"],
        )

    def register_gauge_callback(
        self,
        name: str,
        callback: Callable[[], float],
    ) -> None:
        """Register a callback for a custom gauge"""
        self._custom_gauges[name] = callback

    async def collect(self) -> None:
        """Collect application metrics"""
        for name, callback in self._custom_gauges.items():
            try:
                value = callback()
                gauge = self._registry.get(self._prefixed_name(name))
                if gauge and hasattr(gauge, "set"):
                    gauge.set(value, labels=self._with_labels({}))
            except Exception as e:
                self._logger.debug(
                    "custom_gauge_error",
                    name=name,
                    error=str(e),
                )


class VoiceMetricCollector(MetricCollector):
    """
    Collects Voice AI specific metrics.

    Metrics collected:
    - Active calls and sessions
    - Audio stream quality
    - Transcription latency
    - Agent response times
    - Intent recognition accuracy
    """

    def __init__(
        self,
        config: Optional[CollectorConfig] = None,
        registry: Optional[MetricsRegistry] = None,
    ):
        super().__init__(config or CollectorConfig(prefix="voice"), registry)
        self._setup_metrics()
        self._call_stats: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "voice"

    def _setup_metrics(self) -> None:
        """Setup voice metrics"""
        prefix = self._prefixed_name

        # Calls
        self.active_calls = self._registry.gauge(
            prefix("active_calls"),
            "Number of active calls",
            labels=["agent_id", "organization_id"],
        )
        self.calls_total = self._registry.counter(
            prefix("calls_total"),
            "Total calls",
            labels=["status", "direction", "organization_id"],
        )
        self.call_duration = self._registry.histogram(
            prefix("call_duration_seconds"),
            "Call duration",
            labels=["status"],
            buckets=(10, 30, 60, 120, 300, 600, 1200, 1800, 3600),
        )

        # Audio
        self.audio_packets = self._registry.counter(
            prefix("audio_packets_total"),
            "Total audio packets",
            labels=["direction"],
        )
        self.audio_latency = self._registry.histogram(
            prefix("audio_latency_ms"),
            "Audio latency in milliseconds",
            labels=["type"],
            buckets=(10, 25, 50, 100, 200, 500, 1000),
        )
        self.audio_quality = self._registry.gauge(
            prefix("audio_quality_score"),
            "Audio quality score (0-5)",
            labels=["call_id"],
        )

        # Transcription
        self.transcription_latency = self._registry.histogram(
            prefix("transcription_latency_ms"),
            "Transcription latency",
            labels=["provider"],
            buckets=(50, 100, 200, 500, 1000, 2000, 5000),
        )
        self.transcription_accuracy = self._registry.gauge(
            prefix("transcription_accuracy"),
            "Transcription accuracy (0-1)",
            labels=["provider"],
        )

        # Agent
        self.agent_response_time = self._registry.histogram(
            prefix("agent_response_time_ms"),
            "Agent response time",
            labels=["agent_id", "intent"],
            buckets=(100, 250, 500, 1000, 2000, 5000, 10000),
        )
        self.agent_turns = self._registry.counter(
            prefix("agent_turns_total"),
            "Total agent conversation turns",
            labels=["agent_id"],
        )

        # Intent
        self.intent_detections = self._registry.counter(
            prefix("intent_detections_total"),
            "Total intent detections",
            labels=["intent", "confidence_bucket"],
        )
        self.intent_confidence = self._registry.histogram(
            prefix("intent_confidence"),
            "Intent confidence scores",
            labels=["intent"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )

        # TTS
        self.tts_latency = self._registry.histogram(
            prefix("tts_latency_ms"),
            "Text-to-speech latency",
            labels=["provider", "voice"],
            buckets=(50, 100, 200, 500, 1000, 2000),
        )
        self.tts_characters = self._registry.counter(
            prefix("tts_characters_total"),
            "Total TTS characters processed",
            labels=["provider"],
        )

    def track_call_start(
        self,
        call_id: str,
        agent_id: str,
        organization_id: str,
        direction: str = "inbound",
    ) -> None:
        """Track call start"""
        self._call_stats[call_id] = {
            "start_time": time.time(),
            "agent_id": agent_id,
            "organization_id": organization_id,
            "direction": direction,
        }

        self.active_calls.inc(
            labels=self._with_labels({
                "agent_id": agent_id,
                "organization_id": organization_id,
            })
        )

    def track_call_end(
        self,
        call_id: str,
        status: str = "completed",
    ) -> None:
        """Track call end"""
        stats = self._call_stats.pop(call_id, None)
        if not stats:
            return

        duration = time.time() - stats["start_time"]

        self.active_calls.dec(
            labels=self._with_labels({
                "agent_id": stats["agent_id"],
                "organization_id": stats["organization_id"],
            })
        )

        self.calls_total.inc(
            labels=self._with_labels({
                "status": status,
                "direction": stats["direction"],
                "organization_id": stats["organization_id"],
            })
        )

        self.call_duration.observe(
            duration,
            labels=self._with_labels({"status": status}),
        )

    async def collect(self) -> None:
        """Collect voice metrics"""
        # Active call stats are tracked via track_call_start/end
        # This method can be used for additional periodic collection
        pass


class CollectorManager:
    """
    Manages multiple metric collectors.

    Usage:
        manager = CollectorManager()
        manager.add(SystemMetricCollector())
        manager.add(ProcessMetricCollector())
        await manager.start_all()
    """

    def __init__(self):
        self._collectors: Dict[str, MetricCollector] = {}
        self._logger = structlog.get_logger("collector_manager")

    def add(self, collector: MetricCollector) -> None:
        """Add a collector"""
        if collector.name in self._collectors:
            raise ValueError(f"Collector {collector.name} already registered")
        self._collectors[collector.name] = collector
        self._logger.info("collector_added", name=collector.name)

    def remove(self, name: str) -> Optional[MetricCollector]:
        """Remove a collector"""
        return self._collectors.pop(name, None)

    def get(self, name: str) -> Optional[MetricCollector]:
        """Get a collector by name"""
        return self._collectors.get(name)

    async def start_all(self) -> None:
        """Start all collectors"""
        for collector in self._collectors.values():
            if collector.config.enabled:
                await collector.start()

    async def stop_all(self) -> None:
        """Stop all collectors"""
        for collector in self._collectors.values():
            await collector.stop()

    def list_collectors(self) -> List[str]:
        """List all collector names"""
        return list(self._collectors.keys())

    async def collect_now(self) -> None:
        """Trigger immediate collection from all collectors"""
        for collector in self._collectors.values():
            if collector.config.enabled:
                try:
                    await collector.collect()
                except Exception as e:
                    self._logger.error(
                        "immediate_collection_error",
                        collector=collector.name,
                        error=str(e),
                    )

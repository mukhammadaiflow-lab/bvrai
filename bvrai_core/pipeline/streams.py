"""
Stream Processing Components
============================

Real-time stream processing with windowing, aggregation, and joins.

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)
from uuid import uuid4

import structlog

from bvrai_core.pipeline.engine import DataRecord

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class WindowType(str, Enum):
    """Types of time windows"""

    TUMBLING = "tumbling"      # Fixed, non-overlapping
    SLIDING = "sliding"        # Fixed size, slides by interval
    SESSION = "session"        # Dynamic based on activity
    HOPPING = "hopping"        # Fixed size, hops by interval
    GLOBAL = "global"          # No time boundary


@dataclass
class WindowConfig:
    """Window configuration"""

    type: WindowType = WindowType.TUMBLING
    size_seconds: float = 60.0
    slide_seconds: Optional[float] = None  # For sliding/hopping
    session_gap_seconds: Optional[float] = None  # For session
    max_size: int = 100000
    late_arrival_seconds: float = 10.0


@dataclass
class Window:
    """Represents a time window"""

    id: str = field(default_factory=lambda: str(uuid4()))
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    records: List[DataRecord] = field(default_factory=list)
    key: str = "default"
    closed: bool = False

    @property
    def size(self) -> int:
        return len(self.records)

    def add(self, record: DataRecord) -> None:
        self.records.append(record)

    def close(self) -> None:
        self.end_time = datetime.utcnow()
        self.closed = True


@dataclass
class WindowResult:
    """Result of window aggregation"""

    window_id: str
    key: str
    start_time: datetime
    end_time: datetime
    record_count: int
    result: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class WindowManager:
    """
    Manages time windows for stream processing.
    """

    def __init__(self, config: WindowConfig, key_func: Optional[Callable[[DataRecord], str]] = None):
        self.config = config
        self._key_func = key_func or (lambda r: "default")
        self._windows: Dict[str, List[Window]] = defaultdict(list)
        self._closed_windows: Deque[Window] = deque(maxlen=1000)
        self._logger = structlog.get_logger("window_manager")

    def add(self, record: DataRecord) -> List[Window]:
        """Add a record to appropriate windows"""
        key = self._key_func(record)
        affected_windows = []

        if self.config.type == WindowType.TUMBLING:
            window = self._get_or_create_tumbling_window(key, record.timestamp)
            window.add(record)
            affected_windows.append(window)

        elif self.config.type == WindowType.SLIDING:
            windows = self._get_sliding_windows(key, record.timestamp)
            for window in windows:
                window.add(record)
            affected_windows.extend(windows)

        elif self.config.type == WindowType.SESSION:
            window = self._get_or_create_session_window(key, record.timestamp)
            window.add(record)
            affected_windows.append(window)

        elif self.config.type == WindowType.HOPPING:
            windows = self._get_hopping_windows(key, record.timestamp)
            for window in windows:
                window.add(record)
            affected_windows.extend(windows)

        elif self.config.type == WindowType.GLOBAL:
            window = self._get_or_create_global_window(key)
            window.add(record)
            affected_windows.append(window)

        return affected_windows

    def get_expired_windows(self) -> List[Window]:
        """Get and close expired windows"""
        now = datetime.utcnow()
        expired = []

        for key, windows in self._windows.items():
            for window in windows:
                if window.closed:
                    continue

                should_close = False

                if self.config.type in (WindowType.TUMBLING, WindowType.HOPPING):
                    window_end = window.start_time + timedelta(seconds=self.config.size_seconds)
                    should_close = now > window_end + timedelta(seconds=self.config.late_arrival_seconds)

                elif self.config.type == WindowType.SESSION:
                    if window.records:
                        last_record_time = window.records[-1].timestamp
                        gap = (now - last_record_time).total_seconds()
                        should_close = gap > (self.config.session_gap_seconds or 300)

                if should_close:
                    window.close()
                    expired.append(window)
                    self._closed_windows.append(window)

            # Remove closed windows
            self._windows[key] = [w for w in windows if not w.closed]

        return expired

    def _get_or_create_tumbling_window(self, key: str, timestamp: datetime) -> Window:
        """Get or create a tumbling window"""
        window_start = self._get_window_start(timestamp)

        for window in self._windows[key]:
            if window.start_time == window_start:
                return window

        window = Window(
            start_time=window_start,
            end_time=window_start + timedelta(seconds=self.config.size_seconds),
            key=key
        )
        self._windows[key].append(window)
        return window

    def _get_sliding_windows(self, key: str, timestamp: datetime) -> List[Window]:
        """Get sliding windows that contain the timestamp"""
        windows = []
        slide = self.config.slide_seconds or (self.config.size_seconds / 10)
        size = timedelta(seconds=self.config.size_seconds)

        # Find all windows that should contain this record
        current_start = self._get_window_start(timestamp)

        # Look back for windows that might contain this timestamp
        for i in range(int(self.config.size_seconds / slide) + 1):
            window_start = current_start - timedelta(seconds=slide * i)
            window_end = window_start + size

            if window_start <= timestamp < window_end:
                # Find or create window
                existing = None
                for w in self._windows[key]:
                    if w.start_time == window_start:
                        existing = w
                        break

                if not existing:
                    existing = Window(
                        start_time=window_start,
                        end_time=window_end,
                        key=key
                    )
                    self._windows[key].append(existing)

                windows.append(existing)

        return windows

    def _get_or_create_session_window(self, key: str, timestamp: datetime) -> Window:
        """Get or create a session window"""
        gap = timedelta(seconds=self.config.session_gap_seconds or 300)

        for window in self._windows[key]:
            if window.records:
                last_time = window.records[-1].timestamp
                if timestamp - last_time <= gap:
                    return window

        # Create new session window
        window = Window(start_time=timestamp, key=key)
        self._windows[key].append(window)
        return window

    def _get_hopping_windows(self, key: str, timestamp: datetime) -> List[Window]:
        """Get hopping windows"""
        return self._get_sliding_windows(key, timestamp)

    def _get_or_create_global_window(self, key: str) -> Window:
        """Get or create global window"""
        if not self._windows[key]:
            self._windows[key].append(Window(key=key))
        return self._windows[key][0]

    def _get_window_start(self, timestamp: datetime) -> datetime:
        """Get the start time of the window containing timestamp"""
        epoch = datetime(1970, 1, 1)
        elapsed = (timestamp - epoch).total_seconds()
        window_num = int(elapsed / self.config.size_seconds)
        return epoch + timedelta(seconds=window_num * self.config.size_seconds)


class StreamAggregator:
    """
    Aggregates stream data over windows.

    Usage:
        aggregator = StreamAggregator(
            window_config=WindowConfig(type=WindowType.TUMBLING, size_seconds=60),
            aggregation_func=lambda records: sum(r.value.get("count", 0) for r in records)
        )
    """

    def __init__(
        self,
        window_config: WindowConfig,
        aggregation_func: Callable[[List[DataRecord]], Any],
        key_func: Optional[Callable[[DataRecord], str]] = None
    ):
        self.window_config = window_config
        self._aggregation_func = aggregation_func
        self._window_manager = WindowManager(window_config, key_func)
        self._logger = structlog.get_logger("stream_aggregator")

    def add(self, record: DataRecord) -> Optional[WindowResult]:
        """Add a record and return result if window closes"""
        self._window_manager.add(record)

        # Check for expired windows
        expired = self._window_manager.get_expired_windows()

        if expired:
            # Return first expired window result
            window = expired[0]
            return self._create_result(window)

        return None

    def flush(self) -> List[WindowResult]:
        """Flush all windows and return results"""
        results = []

        for key, windows in self._window_manager._windows.items():
            for window in windows:
                window.close()
                results.append(self._create_result(window))

        self._window_manager._windows.clear()
        return results

    def _create_result(self, window: Window) -> WindowResult:
        """Create aggregation result from window"""
        result = self._aggregation_func(window.records)

        return WindowResult(
            window_id=window.id,
            key=window.key,
            start_time=window.start_time,
            end_time=window.end_time or datetime.utcnow(),
            record_count=len(window.records),
            result=result
        )


class StreamProcessor:
    """
    High-level stream processor with windowing support.

    Usage:
        processor = StreamProcessor()

        processor.window(
            WindowConfig(type=WindowType.TUMBLING, size_seconds=60)
        ).group_by(
            lambda r: r.value.get("agent_id")
        ).aggregate(
            lambda records: {
                "count": len(records),
                "total_duration": sum(r.value.get("duration", 0) for r in records)
            }
        )
    """

    def __init__(self):
        self._window_config: Optional[WindowConfig] = None
        self._key_func: Optional[Callable[[DataRecord], str]] = None
        self._aggregation_func: Optional[Callable[[List[DataRecord]], Any]] = None
        self._filters: List[Callable[[DataRecord], bool]] = []
        self._transforms: List[Callable[[DataRecord], DataRecord]] = []
        self._aggregator: Optional[StreamAggregator] = None

    def window(self, config: WindowConfig) -> "StreamProcessor":
        """Set window configuration"""
        self._window_config = config
        return self

    def group_by(self, key_func: Callable[[DataRecord], str]) -> "StreamProcessor":
        """Set grouping key function"""
        self._key_func = key_func
        return self

    def filter(self, predicate: Callable[[DataRecord], bool]) -> "StreamProcessor":
        """Add a filter"""
        self._filters.append(predicate)
        return self

    def transform(self, func: Callable[[DataRecord], DataRecord]) -> "StreamProcessor":
        """Add a transformation"""
        self._transforms.append(func)
        return self

    def aggregate(self, func: Callable[[List[DataRecord]], Any]) -> "StreamProcessor":
        """Set aggregation function"""
        self._aggregation_func = func
        self._build()
        return self

    def _build(self) -> None:
        """Build the stream processor"""
        if self._window_config and self._aggregation_func:
            self._aggregator = StreamAggregator(
                self._window_config,
                self._aggregation_func,
                self._key_func
            )

    def process(self, record: DataRecord) -> Optional[WindowResult]:
        """Process a single record"""
        # Apply filters
        for f in self._filters:
            if not f(record):
                return None

        # Apply transforms
        for t in self._transforms:
            record = t(record)

        # Add to aggregator
        if self._aggregator:
            return self._aggregator.add(record)

        return None

    def flush(self) -> List[WindowResult]:
        """Flush and return all pending results"""
        if self._aggregator:
            return self._aggregator.flush()
        return []


# =============================================================================
# STREAM JOINS
# =============================================================================


class StreamJoin:
    """
    Join two streams based on key and time window.
    """

    def __init__(
        self,
        left_key_func: Callable[[DataRecord], str],
        right_key_func: Callable[[DataRecord], str],
        window_seconds: float = 60.0,
        join_type: str = "inner"  # inner, left, right, outer
    ):
        self._left_key_func = left_key_func
        self._right_key_func = right_key_func
        self._window_seconds = window_seconds
        self._join_type = join_type
        self._left_buffer: Dict[str, List[Tuple[DataRecord, datetime]]] = defaultdict(list)
        self._right_buffer: Dict[str, List[Tuple[DataRecord, datetime]]] = defaultdict(list)

    def add_left(self, record: DataRecord) -> List[DataRecord]:
        """Add a record from left stream"""
        return self._add_record(record, self._left_key_func, self._left_buffer, self._right_buffer)

    def add_right(self, record: DataRecord) -> List[DataRecord]:
        """Add a record from right stream"""
        return self._add_record(record, self._right_key_func, self._right_buffer, self._left_buffer)

    def _add_record(
        self,
        record: DataRecord,
        key_func: Callable[[DataRecord], str],
        own_buffer: Dict[str, List[Tuple[DataRecord, datetime]]],
        other_buffer: Dict[str, List[Tuple[DataRecord, datetime]]]
    ) -> List[DataRecord]:
        """Add record and attempt joins"""
        key = key_func(record)
        now = datetime.utcnow()

        # Clean old records
        self._clean_buffer(own_buffer)
        self._clean_buffer(other_buffer)

        # Add to buffer
        own_buffer[key].append((record, now))

        # Try to join
        joined = []
        if key in other_buffer:
            for other_record, other_time in other_buffer[key]:
                joined_record = self._create_joined_record(record, other_record)
                joined.append(joined_record)

        return joined

    def _clean_buffer(self, buffer: Dict[str, List[Tuple[DataRecord, datetime]]]) -> None:
        """Remove records outside the window"""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self._window_seconds)

        for key in list(buffer.keys()):
            buffer[key] = [(r, t) for r, t in buffer[key] if t > cutoff]
            if not buffer[key]:
                del buffer[key]

    def _create_joined_record(
        self,
        left: DataRecord,
        right: DataRecord
    ) -> DataRecord:
        """Create a joined record"""
        return DataRecord(
            value={
                "left": left.value,
                "right": right.value
            },
            metadata={
                "left_id": left.id,
                "right_id": right.id,
                "join_type": self._join_type
            },
            timestamp=max(left.timestamp, right.timestamp)
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def tumbling_window(size_seconds: float) -> WindowConfig:
    """Create a tumbling window config"""
    return WindowConfig(type=WindowType.TUMBLING, size_seconds=size_seconds)


def sliding_window(size_seconds: float, slide_seconds: float) -> WindowConfig:
    """Create a sliding window config"""
    return WindowConfig(
        type=WindowType.SLIDING,
        size_seconds=size_seconds,
        slide_seconds=slide_seconds
    )


def session_window(gap_seconds: float) -> WindowConfig:
    """Create a session window config"""
    return WindowConfig(
        type=WindowType.SESSION,
        session_gap_seconds=gap_seconds
    )

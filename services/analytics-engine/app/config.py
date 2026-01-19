"""
Configuration for Analytics Engine Service.
"""

from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MetricType(str, Enum):
    """Types of metrics tracked."""

    # Call metrics
    CALL_VOLUME = "call_volume"
    CALL_DURATION = "call_duration"
    CALL_OUTCOME = "call_outcome"
    CALL_DIRECTION = "call_direction"

    # Conversation metrics
    INTENT = "intent"
    SENTIMENT = "sentiment"
    TOPIC = "topic"
    TURNS = "turns"

    # Performance metrics
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"

    # Business metrics
    CONVERSION = "conversion"
    CSAT = "csat"
    COST = "cost"
    GOAL_COMPLETION = "goal_completion"


class AggregationPeriod(str, Enum):
    """Time periods for aggregation."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class EventType(str, Enum):
    """Types of analytics events."""

    # Call lifecycle
    CALL_STARTED = "call_started"
    CALL_CONNECTED = "call_connected"
    CALL_ENDED = "call_ended"
    CALL_TRANSFERRED = "call_transferred"
    CALL_FAILED = "call_failed"

    # Conversation events
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    INTENT_DETECTED = "intent_detected"
    ENTITY_EXTRACTED = "entity_extracted"
    SENTIMENT_CHANGED = "sentiment_changed"

    # Action events
    ACTION_EXECUTED = "action_executed"
    TRANSFER_INITIATED = "transfer_initiated"
    HANGUP = "hangup"

    # System events
    ERROR = "error"
    TIMEOUT = "timeout"
    LATENCY_RECORDED = "latency_recorded"

    # Business events
    GOAL_ACHIEVED = "goal_achieved"
    CONVERSION = "conversion"
    CSAT_RECORDED = "csat_recorded"


class CallOutcome(str, Enum):
    """Call outcome categories."""

    COMPLETED = "completed"
    TRANSFERRED = "transferred"
    VOICEMAIL = "voicemail"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    FAILED = "failed"
    ABANDONED = "abandoned"


class SentimentCategory(str, Enum):
    """Sentiment categories."""

    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class LatencyComponent(str, Enum):
    """Components for latency tracking."""

    ASR = "asr"
    LLM = "llm"
    TTS = "tts"
    TOTAL = "total"
    NETWORK = "network"
    PROCESSING = "processing"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RetentionPolicy(str, Enum):
    """Data retention policies."""

    RAW_7_DAYS = "raw_7d"
    RAW_30_DAYS = "raw_30d"
    AGGREGATED_90_DAYS = "agg_90d"
    AGGREGATED_1_YEAR = "agg_1y"
    SUMMARY_FOREVER = "summary_forever"


class CollectorConfig(BaseSettings):
    """Event collector configuration."""

    model_config = SettingsConfigDict(env_prefix="COLLECTOR_")

    # Buffering
    buffer_size: int = Field(default=1000, description="Event buffer size")
    flush_interval_ms: int = Field(default=5000, description="Buffer flush interval")
    batch_size: int = Field(default=100, description="Batch size for processing")

    # Validation
    validate_events: bool = Field(default=True, description="Validate incoming events")
    drop_invalid: bool = Field(default=False, description="Drop invalid events")


class ProcessorConfig(BaseSettings):
    """Event processor configuration."""

    model_config = SettingsConfigDict(env_prefix="PROCESSOR_")

    # Parallel processing
    num_workers: int = Field(default=4, description="Number of worker threads")
    queue_size: int = Field(default=10000, description="Processing queue size")

    # Real-time processing
    enable_realtime: bool = Field(default=True, description="Enable real-time processing")
    realtime_window_s: int = Field(default=60, description="Real-time window in seconds")


class AggregatorConfig(BaseSettings):
    """Aggregation configuration."""

    model_config = SettingsConfigDict(env_prefix="AGGREGATOR_")

    # Default periods
    default_periods: List[str] = Field(
        default=["hour", "day", "week", "month"],
        description="Default aggregation periods",
    )

    # Percentiles to calculate
    percentiles: List[float] = Field(
        default=[50, 75, 90, 95, 99],
        description="Percentiles to calculate",
    )

    # Cardinality limits
    max_unique_values: int = Field(
        default=1000,
        description="Max unique values for high-cardinality fields",
    )


class StorageConfig(BaseSettings):
    """Storage configuration."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    # Time series database
    timeseries_url: str = Field(
        default="influxdb://localhost:8086",
        description="Time series database URL",
    )

    # Relational database
    database_url: str = Field(
        default="postgresql://localhost/analytics",
        description="Relational database URL",
    )

    # Redis for real-time
    redis_url: str = Field(
        default="redis://localhost:6379/1",
        description="Redis URL",
    )

    # Retention
    raw_retention_days: int = Field(default=7, description="Raw event retention")
    aggregated_retention_days: int = Field(default=365, description="Aggregated data retention")


class AlertConfig(BaseSettings):
    """Alert configuration."""

    model_config = SettingsConfigDict(env_prefix="ALERT_")

    # Thresholds
    error_rate_threshold: float = Field(default=0.05, description="Error rate alert threshold")
    latency_p95_threshold_ms: int = Field(default=500, description="P95 latency threshold")
    success_rate_threshold: float = Field(default=0.95, description="Success rate threshold")

    # Notifications
    enable_email: bool = Field(default=False, description="Enable email alerts")
    enable_slack: bool = Field(default=False, description="Enable Slack alerts")
    enable_webhook: bool = Field(default=True, description="Enable webhook alerts")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service configuration
    service_name: str = Field(default="analytics-engine", description="Service name")
    host: str = Field(default="0.0.0.0", description="Host to bind")
    port: int = Field(default=8092, ge=1024, le=65535, description="Port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="info", description="Log level")

    # API settings
    api_prefix: str = Field(default="/api/v1", description="API prefix")

    # Sub-configurations
    collector: CollectorConfig = Field(default_factory=CollectorConfig)
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig)
    aggregator: AggregatorConfig = Field(default_factory=AggregatorConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()

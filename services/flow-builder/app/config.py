"""
Configuration for Flow Builder Service.
"""

from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NodeCategory(str, Enum):
    """Node category types."""

    TRIGGER = "trigger"
    ACTION = "action"
    LOGIC = "logic"
    AI = "ai"
    INTEGRATION = "integration"
    UTILITY = "utility"


class NodeType(str, Enum):
    """Available node types."""

    # Triggers
    INCOMING_CALL = "incoming_call"
    OUTBOUND_CALL = "outbound_call"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    EVENT = "event"

    # Actions
    SPEAK = "speak"
    SPEAK_STREAM = "speak_stream"
    PLAY_AUDIO = "play_audio"
    GATHER_INPUT = "gather_input"
    RECORD = "record"
    TRANSFER = "transfer"
    CONFERENCE = "conference"
    HANGUP = "hangup"
    SEND_SMS = "send_sms"
    SEND_EMAIL = "send_email"

    # Logic
    CONDITION = "condition"
    SWITCH = "switch"
    LOOP = "loop"
    WAIT = "wait"
    GOTO = "goto"
    PARALLEL = "parallel"
    TRY_CATCH = "try_catch"

    # AI
    INTENT_DETECTION = "intent_detection"
    ENTITY_EXTRACTION = "entity_extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    LLM_PROMPT = "llm_prompt"
    KNOWLEDGE_QUERY = "knowledge_query"
    VOICE_BIOMETRICS = "voice_biometrics"

    # Integrations
    HTTP_REQUEST = "http_request"
    CRM_LOOKUP = "crm_lookup"
    CRM_UPDATE = "crm_update"
    CALENDAR_CHECK = "calendar_check"
    CALENDAR_BOOK = "calendar_book"
    DATABASE_QUERY = "database_query"
    DATABASE_INSERT = "database_insert"
    QUEUE_PUBLISH = "queue_publish"

    # Utility
    SET_VARIABLE = "set_variable"
    LOG = "log"
    COMMENT = "comment"
    FUNCTION = "function"
    TEMPLATE = "template"


class ConnectionType(str, Enum):
    """Connection/edge types."""

    DEFAULT = "default"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    TRUE = "true"
    FALSE = "false"
    CASE = "case"


class FlowStatus(str, Enum):
    """Flow status values."""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class ExecutionMode(str, Enum):
    """Flow execution modes."""

    PRODUCTION = "production"
    DRY_RUN = "dry_run"
    DEBUG = "debug"
    STEP_THROUGH = "step_through"


class VariableScope(str, Enum):
    """Variable scope."""

    GLOBAL = "global"
    FLOW = "flow"
    SESSION = "session"
    LOCAL = "local"


class DataType(str, Enum):
    """Data types for node inputs/outputs."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    AUDIO = "audio"
    ANY = "any"


class CanvasConfig(BaseSettings):
    """Canvas configuration."""

    model_config = SettingsConfigDict(env_prefix="CANVAS_")

    # Grid settings
    grid_size: int = Field(default=20, description="Grid snap size in pixels")
    enable_snap: bool = Field(default=True, description="Enable grid snapping")

    # Zoom settings
    min_zoom: float = Field(default=0.25, description="Minimum zoom level")
    max_zoom: float = Field(default=2.0, description="Maximum zoom level")
    default_zoom: float = Field(default=1.0, description="Default zoom level")

    # Node settings
    default_node_width: int = Field(default=240, description="Default node width")
    default_node_height: int = Field(default=80, description="Default node height")

    # Validation
    max_nodes_per_flow: int = Field(default=500, description="Max nodes per flow")
    max_connections_per_node: int = Field(default=20, description="Max connections per node")


class StorageConfig(BaseSettings):
    """Storage configuration."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    # Database
    database_url: str = Field(
        default="postgresql://localhost/flow_builder",
        description="Database connection URL",
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for caching",
    )

    # S3/Object storage
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for assets")
    s3_region: str = Field(default="us-east-1", description="S3 region")

    # Versioning
    max_versions: int = Field(default=50, description="Max versions to keep per flow")
    auto_save_interval_s: int = Field(default=30, description="Auto-save interval")


class ExecutionConfig(BaseSettings):
    """Execution configuration."""

    model_config = SettingsConfigDict(env_prefix="EXECUTION_")

    # Timeouts
    node_timeout_ms: int = Field(default=30000, description="Node execution timeout")
    flow_timeout_ms: int = Field(default=300000, description="Flow execution timeout")

    # Limits
    max_loop_iterations: int = Field(default=1000, description="Max loop iterations")
    max_recursion_depth: int = Field(default=50, description="Max recursion depth")
    max_parallel_branches: int = Field(default=10, description="Max parallel branches")

    # Debugging
    enable_debug_logs: bool = Field(default=True, description="Enable debug logging")
    capture_node_timing: bool = Field(default=True, description="Capture node timing")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service configuration
    service_name: str = Field(default="flow-builder", description="Service name")
    host: str = Field(default="0.0.0.0", description="Host to bind")
    port: int = Field(default=8091, ge=1024, le=65535, description="Port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="info", description="Log level")

    # API settings
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    enable_docs: bool = Field(default=True, description="Enable API docs")

    # Authentication
    auth_enabled: bool = Field(default=True, description="Enable authentication")
    jwt_secret: str = Field(default="change-me-in-production", description="JWT secret")

    # Sub-configurations
    canvas: CanvasConfig = Field(default_factory=CanvasConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()

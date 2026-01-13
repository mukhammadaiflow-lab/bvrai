"""
Agent Runtime Base Types Module

This module defines core types and data structures for the agent runtime system.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


class AgentState(str, Enum):
    """States of the agent during execution."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    EXECUTING_TOOL = "executing_tool"
    WAITING = "waiting"
    ERROR = "error"


class ResponseType(str, Enum):
    """Types of agent responses."""

    TEXT = "text"
    FUNCTION_CALL = "function_call"
    END_CALL = "end_call"
    TRANSFER = "transfer"
    HOLD = "hold"
    COLLECT_DTMF = "collect_dtmf"
    PLAY_AUDIO = "play_audio"


class IntentCategory(str, Enum):
    """Categories of detected intents."""

    GREETING = "greeting"
    QUESTION = "question"
    REQUEST = "request"
    COMPLAINT = "complaint"
    CONFIRMATION = "confirmation"
    NEGATION = "negation"
    CLARIFICATION = "clarification"
    SCHEDULING = "scheduling"
    INFORMATION = "information"
    TRANSFER = "transfer"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


class SentimentLevel(str, Enum):
    """Sentiment levels."""

    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


@dataclass
class AgentPersona:
    """Agent persona configuration."""

    # Identity
    name: str
    role: str
    company_name: str

    # Personality
    tone: str = "professional"
    speaking_style: str = "conversational"
    formality_level: str = "business_casual"

    # Industry context
    industry: Optional[str] = None
    specialization: Optional[str] = None

    # Behavior
    greeting_style: str = "warm"
    empathy_level: str = "high"
    assertiveness: str = "moderate"

    # Communication preferences
    preferred_response_length: str = "concise"
    use_filler_words: bool = True
    acknowledge_emotions: bool = True

    # Custom traits
    custom_traits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapabilities:
    """Agent capabilities configuration."""

    # Core capabilities
    can_transfer: bool = True
    can_hold: bool = True
    can_schedule: bool = True
    can_collect_payment: bool = False
    can_access_account: bool = True

    # Knowledge access
    knowledge_base_ids: List[str] = field(default_factory=list)
    can_search_web: bool = False

    # Function calling
    available_functions: List[Dict[str, Any]] = field(default_factory=list)
    function_timeout_seconds: int = 30

    # Transfer targets
    transfer_targets: Dict[str, str] = field(default_factory=dict)

    # Limits
    max_turns_per_call: int = 100
    max_function_calls_per_turn: int = 5


@dataclass
class ConversationContext:
    """Context for the current conversation."""

    # Session info
    session_id: str
    organization_id: str
    agent_id: str

    # Caller info
    caller_id: Optional[str] = None
    caller_name: Optional[str] = None
    caller_account: Optional[Dict[str, Any]] = None

    # Call context
    call_reason: Optional[str] = None
    campaign_id: Optional[str] = None
    is_callback: bool = False

    # Current state
    current_topic: Optional[str] = None
    unresolved_questions: List[str] = field(default_factory=list)
    collected_information: Dict[str, Any] = field(default_factory=dict)

    # Sentiment tracking
    current_sentiment: SentimentLevel = SentimentLevel.NEUTRAL
    sentiment_history: List[Tuple[datetime, SentimentLevel]] = field(default_factory=list)

    # Intent history
    detected_intents: List[Tuple[datetime, IntentCategory]] = field(default_factory=list)

    # Custom variables
    variables: Dict[str, Any] = field(default_factory=dict)

    def update_sentiment(self, sentiment: SentimentLevel) -> None:
        """Update current sentiment."""
        self.sentiment_history.append((datetime.utcnow(), sentiment))
        self.current_sentiment = sentiment

    def add_intent(self, intent: IntentCategory) -> None:
        """Add detected intent."""
        self.detected_intents.append((datetime.utcnow(), intent))

    def set_variable(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(key, default)


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # "system", "user", "assistant", "function"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # For function messages
    function_name: Optional[str] = None
    function_call_id: Optional[str] = None

    # Metadata
    tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class FunctionDefinition:
    """Definition of a callable function."""

    name: str
    description: str
    parameters: Dict[str, Any]

    # Execution config
    timeout_seconds: int = 30
    retry_on_failure: bool = False
    max_retries: int = 2

    # Required permissions
    required_permissions: List[str] = field(default_factory=list)

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class FunctionCall:
    """A function call from the agent."""

    id: str
    name: str
    arguments: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Execution state
    status: str = "pending"  # pending, executing, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class AgentResponse:
    """Response from the agent."""

    type: ResponseType
    content: str = ""

    # For function calls
    function_calls: List[FunctionCall] = field(default_factory=list)

    # For transfers
    transfer_target: Optional[str] = None
    transfer_message: Optional[str] = None

    # For audio
    audio_url: Optional[str] = None

    # Metadata
    thinking_time_ms: float = 0.0
    tokens_used: int = 0
    model_used: str = ""

    # Internal use
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class AgentConfig:
    """Configuration for agent runtime."""

    # Agent identity
    agent_id: str
    name: str
    persona: AgentPersona
    capabilities: AgentCapabilities

    # System prompt
    system_prompt: str
    first_message: Optional[str] = None

    # LLM configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Voice configuration
    voice_provider: str = "elevenlabs"
    voice_id: str = ""
    voice_speed: float = 1.0
    voice_stability: float = 0.5
    voice_similarity: float = 0.75

    # Behavior
    response_timeout_seconds: int = 30
    max_silence_before_prompt_seconds: int = 5
    interruption_sensitivity: float = 0.5

    # Compliance
    record_calls: bool = True
    compliance_mode: Optional[str] = None  # HIPAA, PCI-DSS, etc.

    # Custom configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeMetrics:
    """Metrics for agent runtime."""

    # Response times
    avg_thinking_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    avg_function_execution_time_ms: float = 0.0

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    # Counts
    total_turns: int = 0
    total_function_calls: int = 0
    successful_function_calls: int = 0
    failed_function_calls: int = 0

    # Knowledge retrieval
    knowledge_queries: int = 0
    knowledge_hits: int = 0

    def add_turn_metrics(
        self,
        thinking_time_ms: float,
        response_time_ms: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Add metrics for a turn."""
        self.total_turns += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens

        # Update averages
        n = self.total_turns
        self.avg_thinking_time_ms = (
            (self.avg_thinking_time_ms * (n - 1) + thinking_time_ms) / n
        )
        self.avg_response_time_ms = (
            (self.avg_response_time_ms * (n - 1) + response_time_ms) / n
        )


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge from the knowledge base."""

    id: str
    content: str
    source: str
    score: float = 0.0

    # Metadata
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None

    # Embedding info
    embedding_model: Optional[str] = None
    chunk_index: int = 0


@dataclass
class KnowledgeQuery:
    """A query to the knowledge base."""

    query: str
    top_k: int = 5
    min_score: float = 0.5
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeResult:
    """Result from knowledge base query."""

    chunks: List[KnowledgeChunk]
    query_time_ms: float = 0.0
    total_results: int = 0


class RuntimeError(Exception):
    """Base exception for runtime errors."""
    pass


class AgentExecutionError(RuntimeError):
    """Error during agent execution."""
    pass


class FunctionExecutionError(RuntimeError):
    """Error during function execution."""
    pass


class KnowledgeRetrievalError(RuntimeError):
    """Error during knowledge retrieval."""
    pass


class TokenLimitExceededError(RuntimeError):
    """Token limit exceeded."""
    pass


__all__ = [
    "AgentState",
    "ResponseType",
    "IntentCategory",
    "SentimentLevel",
    "AgentPersona",
    "AgentCapabilities",
    "ConversationContext",
    "Message",
    "FunctionDefinition",
    "FunctionCall",
    "AgentResponse",
    "AgentConfig",
    "RuntimeMetrics",
    "KnowledgeChunk",
    "KnowledgeQuery",
    "KnowledgeResult",
    "RuntimeError",
    "AgentExecutionError",
    "FunctionExecutionError",
    "KnowledgeRetrievalError",
    "TokenLimitExceededError",
]

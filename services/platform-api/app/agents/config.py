"""
Agent Configuration

Core agent configuration and types:
- Agent definition
- Configuration management
- Capability definitions
"""

from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class AgentStatus(str, Enum):
    """Agent status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    TESTING = "testing"


class AgentType(str, Enum):
    """Agent types."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BOTH = "both"
    IVR = "ivr"
    ASSISTANT = "assistant"


class AgentCapability(str, Enum):
    """Agent capabilities."""
    # Communication
    VOICE = "voice"
    TEXT = "text"
    DTMF = "dtmf"

    # Language
    SPEECH_TO_TEXT = "stt"
    TEXT_TO_SPEECH = "tts"
    LANGUAGE_DETECTION = "language_detection"
    TRANSLATION = "translation"

    # Understanding
    INTENT_DETECTION = "intent_detection"
    ENTITY_EXTRACTION = "entity_extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

    # Actions
    FUNCTION_CALLING = "function_calling"
    KNOWLEDGE_BASE = "knowledge_base"
    LIVE_TRANSFER = "live_transfer"
    CALL_RECORDING = "call_recording"

    # Advanced
    INTERRUPTION_HANDLING = "interruption_handling"
    MULTI_TURN = "multi_turn"
    CONTEXT_CARRYOVER = "context_carryover"


@dataclass
class VoiceSettings:
    """Voice configuration settings."""
    # Provider
    provider: str = "elevenlabs"  # elevenlabs, azure, google, openai
    voice_id: str = ""
    model_id: str = ""

    # Voice characteristics
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True

    # Speed and pitch
    speaking_rate: float = 1.0
    pitch: float = 0.0

    # Language
    language: str = "en-US"
    accent: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "voice_id": self.voice_id,
            "model_id": self.model_id,
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "speaking_rate": self.speaking_rate,
            "pitch": self.pitch,
            "language": self.language,
        }


@dataclass
class TranscriptionSettings:
    """Speech-to-text settings."""
    # Provider
    provider: str = "deepgram"  # deepgram, google, azure, whisper
    model: str = "nova-2"

    # Language
    language: str = "en-US"
    detect_language: bool = False

    # Features
    punctuate: bool = True
    profanity_filter: bool = False
    diarize: bool = False
    smart_format: bool = True

    # Performance
    interim_results: bool = True
    endpointing_ms: int = 300

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "language": self.language,
            "detect_language": self.detect_language,
            "punctuate": self.punctuate,
            "interim_results": self.interim_results,
        }


@dataclass
class LLMSettings:
    """Language model settings."""
    # Provider
    provider: str = "openai"  # openai, anthropic, google, azure
    model: str = "gpt-4"

    # Parameters
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Context
    max_context_messages: int = 50
    context_window: int = 8192

    # Features
    streaming: bool = True
    function_calling: bool = True
    json_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "function_calling": self.function_calling,
        }


@dataclass
class InterruptionConfig:
    """Interruption handling configuration."""
    enabled: bool = True

    # Thresholds
    min_words_before_interrupt: int = 5
    silence_threshold_ms: int = 500
    confidence_threshold: float = 0.7

    # Behavior
    allow_user_interrupt: bool = True
    allow_barge_in: bool = True
    pause_on_interrupt: bool = True
    resume_after_interrupt: bool = True

    # Timing
    interrupt_detection_ms: int = 200
    resume_delay_ms: int = 500


@dataclass
class SilenceConfig:
    """Silence handling configuration."""
    # Detection
    initial_silence_timeout_ms: int = 10000
    max_silence_timeout_ms: int = 5000
    end_of_speech_silence_ms: int = 1000

    # Responses
    silence_prompts: List[str] = field(default_factory=lambda: [
        "Are you still there?",
        "I didn't hear anything. Could you please continue?",
    ])
    max_silence_prompts: int = 2

    # Actions
    action_on_max_silence: str = "transfer"  # transfer, hangup, retry


@dataclass
class ErrorConfig:
    """Error handling configuration."""
    # Retry settings
    max_retries: int = 3
    retry_delay_ms: int = 1000

    # Error responses
    general_error_message: str = "I apologize, but I encountered an issue. Let me try again."
    max_errors_before_transfer: int = 3

    # Fallback
    fallback_action: str = "transfer"  # transfer, hangup, message


@dataclass
class AgentConfig:
    """Complete agent configuration."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tenant_id: str = ""

    # Type and status
    agent_type: AgentType = AgentType.INBOUND
    status: AgentStatus = AgentStatus.DRAFT

    # Capabilities
    capabilities: Set[AgentCapability] = field(default_factory=set)

    # Core settings
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)

    # Behavior settings
    interruption: InterruptionConfig = field(default_factory=InterruptionConfig)
    silence: SilenceConfig = field(default_factory=SilenceConfig)
    error: ErrorConfig = field(default_factory=ErrorConfig)

    # Prompts
    system_prompt: str = ""
    first_message: str = ""
    end_call_message: str = ""

    # Knowledge
    knowledge_base_ids: List[str] = field(default_factory=list)

    # Functions
    function_ids: List[str] = field(default_factory=list)

    # Webhooks
    webhook_url: Optional[str] = None
    webhook_events: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has capability."""
        return capability in self.capabilities

    def add_capability(self, capability: AgentCapability) -> None:
        """Add capability."""
        self.capabilities.add(capability)

    def remove_capability(self, capability: AgentCapability) -> None:
        """Remove capability."""
        self.capabilities.discard(capability)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "tenant_id": self.tenant_id,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "capabilities": [c.value for c in self.capabilities],
            "voice": self.voice.to_dict(),
            "transcription": self.transcription.to_dict(),
            "llm": self.llm.to_dict(),
            "system_prompt": self.system_prompt,
            "first_message": self.first_message,
            "knowledge_base_ids": self.knowledge_base_ids,
            "function_ids": self.function_ids,
            "webhook_url": self.webhook_url,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create from dictionary."""
        config = cls(
            agent_id=data.get("agent_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            tenant_id=data.get("tenant_id", ""),
            agent_type=AgentType(data.get("agent_type", "inbound")),
            status=AgentStatus(data.get("status", "draft")),
            system_prompt=data.get("system_prompt", ""),
            first_message=data.get("first_message", ""),
            knowledge_base_ids=data.get("knowledge_base_ids", []),
            function_ids=data.get("function_ids", []),
            webhook_url=data.get("webhook_url"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )

        # Parse capabilities
        capabilities = data.get("capabilities", [])
        for cap in capabilities:
            try:
                config.capabilities.add(AgentCapability(cap))
            except ValueError:
                pass

        return config


class AgentValidator:
    """Validates agent configuration."""

    @staticmethod
    def validate(config: AgentConfig) -> List[str]:
        """Validate agent configuration. Returns list of errors."""
        errors = []

        # Required fields
        if not config.name:
            errors.append("Agent name is required")

        if not config.tenant_id:
            errors.append("Tenant ID is required")

        if not config.system_prompt:
            errors.append("System prompt is required")

        # Voice settings
        if AgentCapability.VOICE in config.capabilities:
            if not config.voice.voice_id:
                errors.append("Voice ID is required for voice capability")

        # LLM settings
        if config.llm.temperature < 0 or config.llm.temperature > 2:
            errors.append("LLM temperature must be between 0 and 2")

        if config.llm.max_tokens < 1:
            errors.append("LLM max_tokens must be positive")

        # Transcription
        if AgentCapability.SPEECH_TO_TEXT in config.capabilities:
            if not config.transcription.language:
                errors.append("Transcription language is required")

        return errors

    @staticmethod
    def is_valid(config: AgentConfig) -> bool:
        """Check if configuration is valid."""
        return len(AgentValidator.validate(config)) == 0

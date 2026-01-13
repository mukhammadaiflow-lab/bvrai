"""
Voice Configuration Dashboard Module

This module provides a comprehensive voice configuration system that allows
users to configure STT, TTS, and voice settings for AI agents. This is
designed to be better than competitors like Vapi, with more flexibility,
more providers, and advanced features.

Key Features:
- Multi-provider STT support (Deepgram, Google, Azure, AssemblyAI, Whisper, etc.)
- Multi-provider TTS support (ElevenLabs, PlayHT, Azure, Google, OpenAI, etc.)
- Voice library with categorization
- Custom voice ID support
- Model selection per provider
- Voice preview and testing
- Configuration presets
- A/B testing support for voice configurations
"""

import uuid
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# =============================================================================
# Provider Enums
# =============================================================================


class STTProvider(str, Enum):
    """Speech-to-Text providers."""

    DEEPGRAM = "deepgram"
    GOOGLE = "google"
    AZURE = "azure"
    ASSEMBLYAI = "assemblyai"
    WHISPER = "whisper"
    OPENAI = "openai"
    AWS_TRANSCRIBE = "aws_transcribe"
    REV_AI = "rev_ai"
    SPEECHMATICS = "speechmatics"


class TTSProvider(str, Enum):
    """Text-to-Speech providers."""

    ELEVENLABS = "elevenlabs"
    PLAYHT = "playht"
    AZURE = "azure"
    GOOGLE = "google"
    OPENAI = "openai"
    AWS_POLLY = "aws_polly"
    CARTESIA = "cartesia"
    RIME = "rime"
    DEEPGRAM = "deepgram"
    WELLSAID = "wellsaid"
    MURF = "murf"


class VoiceGender(str, Enum):
    """Voice gender classification."""

    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceStyle(str, Enum):
    """Voice style/character."""

    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    CALM = "calm"
    ENERGETIC = "energetic"
    WARM = "warm"
    NARRATIVE = "narrative"
    NEWSCAST = "newscast"
    CUSTOMER_SERVICE = "customer_service"


class VoiceAge(str, Enum):
    """Voice age range."""

    YOUNG = "young"  # 18-30
    MIDDLE = "middle"  # 30-50
    MATURE = "mature"  # 50+


class LanguageCode(str, Enum):
    """Supported language codes."""

    EN_US = "en-US"
    EN_GB = "en-GB"
    EN_AU = "en-AU"
    ES_ES = "es-ES"
    ES_MX = "es-MX"
    FR_FR = "fr-FR"
    FR_CA = "fr-CA"
    DE_DE = "de-DE"
    IT_IT = "it-IT"
    PT_BR = "pt-BR"
    PT_PT = "pt-PT"
    JA_JP = "ja-JP"
    KO_KR = "ko-KR"
    ZH_CN = "zh-CN"
    ZH_TW = "zh-TW"
    HI_IN = "hi-IN"
    AR_SA = "ar-SA"
    RU_RU = "ru-RU"
    NL_NL = "nl-NL"
    PL_PL = "pl-PL"
    TR_TR = "tr-TR"
    VI_VN = "vi-VN"
    TH_TH = "th-TH"
    ID_ID = "id-ID"
    MS_MY = "ms-MY"
    FIL_PH = "fil-PH"
    SV_SE = "sv-SE"
    DA_DK = "da-DK"
    NO_NO = "no-NO"
    FI_FI = "fi-FI"
    CS_CZ = "cs-CZ"
    EL_GR = "el-GR"
    HE_IL = "he-IL"
    UK_UA = "uk-UA"
    RO_RO = "ro-RO"
    HU_HU = "hu-HU"
    SK_SK = "sk-SK"
    BG_BG = "bg-BG"
    HR_HR = "hr-HR"


# =============================================================================
# STT Configuration Types
# =============================================================================


@dataclass
class STTModelConfig:
    """Configuration for a specific STT model."""

    model_id: str
    display_name: str
    provider: STTProvider

    # Capabilities
    languages: List[LanguageCode] = field(default_factory=list)
    supports_streaming: bool = True
    supports_diarization: bool = False
    supports_punctuation: bool = True
    supports_profanity_filter: bool = False
    supports_word_timestamps: bool = False

    # Performance characteristics
    latency_ms_estimate: int = 200
    accuracy_tier: str = "standard"  # standard, enhanced, premium

    # Pricing (per minute)
    cost_per_minute: float = 0.0

    # Metadata
    description: str = ""
    recommended_for: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "provider": self.provider.value,
            "languages": [l.value for l in self.languages],
            "supports_streaming": self.supports_streaming,
            "supports_diarization": self.supports_diarization,
            "supports_punctuation": self.supports_punctuation,
            "supports_word_timestamps": self.supports_word_timestamps,
            "latency_ms_estimate": self.latency_ms_estimate,
            "accuracy_tier": self.accuracy_tier,
            "cost_per_minute": self.cost_per_minute,
            "description": self.description,
            "recommended_for": self.recommended_for,
        }


@dataclass
class STTProviderConfig:
    """Configuration for an STT provider."""

    provider: STTProvider
    display_name: str

    # API Configuration
    api_key: str = ""
    api_endpoint: Optional[str] = None
    region: Optional[str] = None

    # Available models
    models: List[STTModelConfig] = field(default_factory=list)
    default_model_id: str = ""

    # Provider settings
    is_enabled: bool = True
    priority: int = 0  # Lower = higher priority

    # Rate limiting
    max_concurrent_requests: int = 100
    requests_per_minute: int = 1000

    # Health status
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None
    error_rate: float = 0.0

    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "provider": self.provider.value,
            "display_name": self.display_name,
            "api_endpoint": self.api_endpoint,
            "region": self.region,
            "models": [m.to_dict() for m in self.models],
            "default_model_id": self.default_model_id,
            "is_enabled": self.is_enabled,
            "priority": self.priority,
            "max_concurrent_requests": self.max_concurrent_requests,
            "requests_per_minute": self.requests_per_minute,
            "is_healthy": self.is_healthy,
            "error_rate": self.error_rate,
        }
        if include_secrets:
            data["api_key"] = self.api_key
        return data


@dataclass
class STTConfiguration:
    """Complete STT configuration for an agent."""

    id: str
    organization_id: str
    name: str

    # Provider and model selection
    provider: STTProvider = STTProvider.DEEPGRAM
    model_id: str = "nova-2"

    # Language settings
    language: LanguageCode = LanguageCode.EN_US
    detect_language: bool = False
    alternative_languages: List[LanguageCode] = field(default_factory=list)

    # Transcription settings
    enable_punctuation: bool = True
    enable_profanity_filter: bool = False
    enable_diarization: bool = False
    enable_word_timestamps: bool = False

    # Audio processing
    sample_rate: int = 16000
    encoding: str = "linear16"  # linear16, mulaw, flac
    channels: int = 1

    # Streaming settings
    interim_results: bool = True
    endpointing_ms: int = 500
    utterance_end_ms: int = 1000

    # Keywords and context
    keywords: List[str] = field(default_factory=list)
    keyword_boost: float = 1.0
    custom_vocabulary: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"stt_config_{uuid.uuid4().hex[:16]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "provider": self.provider.value,
            "model_id": self.model_id,
            "language": self.language.value,
            "detect_language": self.detect_language,
            "alternative_languages": [l.value for l in self.alternative_languages],
            "enable_punctuation": self.enable_punctuation,
            "enable_profanity_filter": self.enable_profanity_filter,
            "enable_diarization": self.enable_diarization,
            "enable_word_timestamps": self.enable_word_timestamps,
            "sample_rate": self.sample_rate,
            "encoding": self.encoding,
            "channels": self.channels,
            "interim_results": self.interim_results,
            "endpointing_ms": self.endpointing_ms,
            "utterance_end_ms": self.utterance_end_ms,
            "keywords": self.keywords,
            "keyword_boost": self.keyword_boost,
            "custom_vocabulary": self.custom_vocabulary,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# TTS Configuration Types
# =============================================================================


@dataclass
class Voice:
    """A voice available for TTS."""

    id: str
    provider: TTSProvider
    voice_id: str  # Provider-specific voice ID

    # Display information
    name: str
    display_name: str

    # Voice characteristics
    gender: VoiceGender = VoiceGender.NEUTRAL
    age: VoiceAge = VoiceAge.MIDDLE
    style: VoiceStyle = VoiceStyle.CONVERSATIONAL
    language: LanguageCode = LanguageCode.EN_US
    accent: str = ""

    # Supported features
    supports_ssml: bool = True
    supports_streaming: bool = True
    supports_cloning: bool = False

    # Audio characteristics
    sample_rate: int = 24000

    # Preview
    preview_url: Optional[str] = None
    preview_text: str = "Hello, this is a sample of my voice."

    # Categories for organization
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Usage stats
    is_premium: bool = False
    is_custom: bool = False
    is_public: bool = True

    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"voice_{uuid.uuid4().hex[:16]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "provider": self.provider.value,
            "voice_id": self.voice_id,
            "name": self.name,
            "display_name": self.display_name,
            "gender": self.gender.value,
            "age": self.age.value,
            "style": self.style.value,
            "language": self.language.value,
            "accent": self.accent,
            "supports_ssml": self.supports_ssml,
            "supports_streaming": self.supports_streaming,
            "supports_cloning": self.supports_cloning,
            "sample_rate": self.sample_rate,
            "preview_url": self.preview_url,
            "categories": self.categories,
            "tags": self.tags,
            "is_premium": self.is_premium,
            "is_custom": self.is_custom,
            "is_public": self.is_public,
            "description": self.description,
        }


@dataclass
class TTSModelConfig:
    """Configuration for a specific TTS model."""

    model_id: str
    display_name: str
    provider: TTSProvider

    # Model characteristics
    quality_tier: str = "standard"  # standard, enhanced, premium, turbo
    latency_tier: str = "standard"  # low, standard, high

    # Supported features
    supports_ssml: bool = True
    supports_streaming: bool = True
    supports_emotions: bool = False
    supports_speed_control: bool = True
    supports_pitch_control: bool = True

    # Performance
    latency_ms_estimate: int = 200

    # Pricing
    cost_per_1k_chars: float = 0.0

    # Voice compatibility
    compatible_voices: List[str] = field(default_factory=list)  # Empty = all voices

    # Metadata
    description: str = ""
    recommended_for: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "provider": self.provider.value,
            "quality_tier": self.quality_tier,
            "latency_tier": self.latency_tier,
            "supports_ssml": self.supports_ssml,
            "supports_streaming": self.supports_streaming,
            "supports_emotions": self.supports_emotions,
            "supports_speed_control": self.supports_speed_control,
            "supports_pitch_control": self.supports_pitch_control,
            "latency_ms_estimate": self.latency_ms_estimate,
            "cost_per_1k_chars": self.cost_per_1k_chars,
            "description": self.description,
            "recommended_for": self.recommended_for,
        }


@dataclass
class TTSProviderConfig:
    """Configuration for a TTS provider."""

    provider: TTSProvider
    display_name: str

    # API Configuration
    api_key: str = ""
    api_endpoint: Optional[str] = None
    region: Optional[str] = None

    # Available models and voices
    models: List[TTSModelConfig] = field(default_factory=list)
    voices: List[Voice] = field(default_factory=list)
    default_model_id: str = ""
    default_voice_id: str = ""

    # Provider settings
    is_enabled: bool = True
    priority: int = 0

    # Rate limiting
    max_concurrent_requests: int = 100
    requests_per_minute: int = 1000
    characters_per_minute: int = 100000

    # Health status
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None
    error_rate: float = 0.0

    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "provider": self.provider.value,
            "display_name": self.display_name,
            "api_endpoint": self.api_endpoint,
            "region": self.region,
            "models": [m.to_dict() for m in self.models],
            "voices": [v.to_dict() for v in self.voices],
            "default_model_id": self.default_model_id,
            "default_voice_id": self.default_voice_id,
            "is_enabled": self.is_enabled,
            "priority": self.priority,
            "is_healthy": self.is_healthy,
            "error_rate": self.error_rate,
        }
        if include_secrets:
            data["api_key"] = self.api_key
        return data


@dataclass
class VoiceSettings:
    """Voice synthesis settings."""

    # Speed and pitch
    speed: float = 1.0  # 0.25 to 4.0
    pitch: float = 1.0  # 0.5 to 2.0

    # Volume
    volume: float = 1.0  # 0.0 to 2.0
    volume_gain_db: float = 0.0  # -20 to 20

    # ElevenLabs specific
    stability: float = 0.5  # 0.0 to 1.0
    similarity_boost: float = 0.75  # 0.0 to 1.0
    style: float = 0.0  # 0.0 to 1.0
    use_speaker_boost: bool = True

    # Emotion/style (for supported providers)
    emotion: Optional[str] = None  # happy, sad, angry, etc.
    speaking_style: Optional[str] = None  # chat, newscast, etc.

    # Audio output
    output_format: str = "mp3_44100_128"  # Provider-specific
    sample_rate: int = 24000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "speed": self.speed,
            "pitch": self.pitch,
            "volume": self.volume,
            "volume_gain_db": self.volume_gain_db,
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost,
            "emotion": self.emotion,
            "speaking_style": self.speaking_style,
            "output_format": self.output_format,
            "sample_rate": self.sample_rate,
        }


@dataclass
class TTSConfiguration:
    """Complete TTS configuration for an agent."""

    id: str
    organization_id: str
    name: str

    # Provider and model selection
    provider: TTSProvider = TTSProvider.ELEVENLABS
    model_id: str = "eleven_turbo_v2_5"

    # Voice selection
    voice_id: str = ""
    custom_voice_id: Optional[str] = None  # For manually added voice IDs

    # Voice settings
    settings: VoiceSettings = field(default_factory=VoiceSettings)

    # Language
    language: LanguageCode = LanguageCode.EN_US

    # SSML settings
    enable_ssml: bool = False
    ssml_template: Optional[str] = None

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600

    # Fallback
    fallback_provider: Optional[TTSProvider] = None
    fallback_voice_id: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"tts_config_{uuid.uuid4().hex[:16]}"

    @property
    def effective_voice_id(self) -> str:
        """Get the effective voice ID (custom or standard)."""
        return self.custom_voice_id or self.voice_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "provider": self.provider.value,
            "model_id": self.model_id,
            "voice_id": self.voice_id,
            "custom_voice_id": self.custom_voice_id,
            "effective_voice_id": self.effective_voice_id,
            "settings": self.settings.to_dict(),
            "language": self.language.value,
            "enable_ssml": self.enable_ssml,
            "ssml_template": self.ssml_template,
            "enable_cache": self.enable_cache,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "fallback_provider": self.fallback_provider.value if self.fallback_provider else None,
            "fallback_voice_id": self.fallback_voice_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# Complete Voice Configuration
# =============================================================================


@dataclass
class VoiceConfiguration:
    """
    Complete voice configuration for an AI agent.

    This is the main configuration object that combines STT, TTS,
    and all voice-related settings for an agent.
    """

    id: str
    organization_id: str
    agent_id: str
    name: str

    # STT Configuration
    stt: STTConfiguration = field(default_factory=lambda: STTConfiguration(
        id="", organization_id="", name=""
    ))

    # TTS Configuration
    tts: TTSConfiguration = field(default_factory=lambda: TTSConfiguration(
        id="", organization_id="", name=""
    ))

    # Voice Activity Detection settings
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_max_speech_duration_ms: int = 30000
    vad_silence_duration_ms: int = 500
    vad_padding_ms: int = 200

    # Turn-taking settings
    turn_detection_enabled: bool = True
    turn_end_silence_ms: int = 700
    interruption_threshold: float = 0.3
    allow_interruption: bool = True

    # Backchanneling settings (like "uh-huh", "okay")
    backchanneling_enabled: bool = False
    backchannel_words: List[str] = field(default_factory=lambda: [
        "uh-huh", "okay", "I see", "right", "mm-hmm"
    ])
    backchannel_frequency: float = 0.1  # Probability of backchanneling

    # Filler words during processing
    filler_enabled: bool = False
    filler_words: List[str] = field(default_factory=lambda: [
        "let me check", "one moment", "hmm"
    ])
    filler_delay_ms: int = 1500  # When to start filler

    # Audio processing
    noise_suppression: bool = True
    echo_cancellation: bool = True
    auto_gain_control: bool = True

    # Latency optimization
    optimize_latency: bool = True
    target_latency_ms: int = 500

    # Status
    is_active: bool = True
    is_default: bool = False

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"voice_config_{uuid.uuid4().hex[:14]}"

        # Propagate organization_id to child configs
        if self.stt and not self.stt.organization_id:
            self.stt.organization_id = self.organization_id
        if self.tts and not self.tts.organization_id:
            self.tts.organization_id = self.organization_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "agent_id": self.agent_id,
            "name": self.name,
            "stt": self.stt.to_dict() if self.stt else None,
            "tts": self.tts.to_dict() if self.tts else None,
            "vad_enabled": self.vad_enabled,
            "vad_threshold": self.vad_threshold,
            "vad_min_speech_duration_ms": self.vad_min_speech_duration_ms,
            "vad_max_speech_duration_ms": self.vad_max_speech_duration_ms,
            "vad_silence_duration_ms": self.vad_silence_duration_ms,
            "vad_padding_ms": self.vad_padding_ms,
            "turn_detection_enabled": self.turn_detection_enabled,
            "turn_end_silence_ms": self.turn_end_silence_ms,
            "interruption_threshold": self.interruption_threshold,
            "allow_interruption": self.allow_interruption,
            "backchanneling_enabled": self.backchanneling_enabled,
            "backchannel_words": self.backchannel_words,
            "backchannel_frequency": self.backchannel_frequency,
            "filler_enabled": self.filler_enabled,
            "filler_words": self.filler_words,
            "filler_delay_ms": self.filler_delay_ms,
            "noise_suppression": self.noise_suppression,
            "echo_cancellation": self.echo_cancellation,
            "auto_gain_control": self.auto_gain_control,
            "optimize_latency": self.optimize_latency,
            "target_latency_ms": self.target_latency_ms,
            "is_active": self.is_active,
            "is_default": self.is_default,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceConfiguration":
        """Create from dictionary."""
        stt_data = data.get("stt", {})
        tts_data = data.get("tts", {})

        return cls(
            id=data.get("id", ""),
            organization_id=data["organization_id"],
            agent_id=data["agent_id"],
            name=data["name"],
            stt=STTConfiguration(
                id=stt_data.get("id", ""),
                organization_id=stt_data.get("organization_id", data["organization_id"]),
                name=stt_data.get("name", ""),
                provider=STTProvider(stt_data.get("provider", "deepgram")),
                model_id=stt_data.get("model_id", "nova-2"),
                language=LanguageCode(stt_data.get("language", "en-US")),
            ) if stt_data else None,
            tts=TTSConfiguration(
                id=tts_data.get("id", ""),
                organization_id=tts_data.get("organization_id", data["organization_id"]),
                name=tts_data.get("name", ""),
                provider=TTSProvider(tts_data.get("provider", "elevenlabs")),
                model_id=tts_data.get("model_id", "eleven_turbo_v2_5"),
                voice_id=tts_data.get("voice_id", ""),
                custom_voice_id=tts_data.get("custom_voice_id"),
            ) if tts_data else None,
            vad_enabled=data.get("vad_enabled", True),
            vad_threshold=data.get("vad_threshold", 0.5),
            turn_detection_enabled=data.get("turn_detection_enabled", True),
            turn_end_silence_ms=data.get("turn_end_silence_ms", 700),
            allow_interruption=data.get("allow_interruption", True),
            backchanneling_enabled=data.get("backchanneling_enabled", False),
            filler_enabled=data.get("filler_enabled", False),
            noise_suppression=data.get("noise_suppression", True),
            optimize_latency=data.get("optimize_latency", True),
            target_latency_ms=data.get("target_latency_ms", 500),
            is_active=data.get("is_active", True),
            is_default=data.get("is_default", False),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )


# =============================================================================
# Configuration Preset
# =============================================================================


@dataclass
class VoiceConfigurationPreset:
    """
    A preset configuration that can be applied quickly.

    Presets provide quick setup for common use cases like:
    - Low latency (for conversational AI)
    - High quality (for premium experiences)
    - Cost optimized (for high volume)
    - Multilingual support
    """

    id: str
    name: str
    description: str

    # Category
    category: str = "general"  # general, conversational, high_quality, cost_optimized

    # Preset configuration
    stt_provider: STTProvider = STTProvider.DEEPGRAM
    stt_model_id: str = "nova-2"
    tts_provider: TTSProvider = TTSProvider.ELEVENLABS
    tts_model_id: str = "eleven_turbo_v2_5"

    # Voice settings
    voice_settings: VoiceSettings = field(default_factory=VoiceSettings)

    # Feature flags
    vad_enabled: bool = True
    turn_detection_enabled: bool = True
    allow_interruption: bool = True
    backchanneling_enabled: bool = False
    filler_enabled: bool = False
    noise_suppression: bool = True

    # Target metrics
    target_latency_ms: int = 500
    estimated_cost_per_minute: float = 0.10

    # Use case tags
    recommended_for: List[str] = field(default_factory=list)

    # Status
    is_system: bool = False  # System presets can't be deleted
    is_public: bool = True

    def __post_init__(self):
        if not self.id:
            self.id = f"preset_{uuid.uuid4().hex[:16]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "stt_provider": self.stt_provider.value,
            "stt_model_id": self.stt_model_id,
            "tts_provider": self.tts_provider.value,
            "tts_model_id": self.tts_model_id,
            "voice_settings": self.voice_settings.to_dict(),
            "vad_enabled": self.vad_enabled,
            "turn_detection_enabled": self.turn_detection_enabled,
            "allow_interruption": self.allow_interruption,
            "backchanneling_enabled": self.backchanneling_enabled,
            "filler_enabled": self.filler_enabled,
            "noise_suppression": self.noise_suppression,
            "target_latency_ms": self.target_latency_ms,
            "estimated_cost_per_minute": self.estimated_cost_per_minute,
            "recommended_for": self.recommended_for,
            "is_system": self.is_system,
            "is_public": self.is_public,
        }

    def apply_to(self, config: VoiceConfiguration) -> VoiceConfiguration:
        """Apply preset to a configuration."""
        config.stt.provider = self.stt_provider
        config.stt.model_id = self.stt_model_id
        config.tts.provider = self.tts_provider
        config.tts.model_id = self.tts_model_id
        config.tts.settings = VoiceSettings(**self.voice_settings.to_dict())
        config.vad_enabled = self.vad_enabled
        config.turn_detection_enabled = self.turn_detection_enabled
        config.allow_interruption = self.allow_interruption
        config.backchanneling_enabled = self.backchanneling_enabled
        config.filler_enabled = self.filler_enabled
        config.noise_suppression = self.noise_suppression
        config.target_latency_ms = self.target_latency_ms
        config.updated_at = datetime.utcnow()
        return config


# =============================================================================
# Exceptions
# =============================================================================


class VoiceConfigurationError(Exception):
    """Base exception for voice configuration errors."""
    pass


class ProviderNotFoundError(VoiceConfigurationError):
    """Provider not found or not configured."""
    pass


class VoiceNotFoundError(VoiceConfigurationError):
    """Voice not found."""
    pass


class ModelNotFoundError(VoiceConfigurationError):
    """Model not found for provider."""
    pass


class InvalidConfigurationError(VoiceConfigurationError):
    """Configuration validation failed."""
    pass


class ProviderConnectionError(VoiceConfigurationError):
    """Could not connect to provider."""
    pass


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Provider enums
    "STTProvider",
    "TTSProvider",
    # Voice enums
    "VoiceGender",
    "VoiceStyle",
    "VoiceAge",
    "LanguageCode",
    # STT types
    "STTModelConfig",
    "STTProviderConfig",
    "STTConfiguration",
    # TTS types
    "Voice",
    "TTSModelConfig",
    "TTSProviderConfig",
    "VoiceSettings",
    "TTSConfiguration",
    # Main configuration
    "VoiceConfiguration",
    "VoiceConfigurationPreset",
    # Exceptions
    "VoiceConfigurationError",
    "ProviderNotFoundError",
    "VoiceNotFoundError",
    "ModelNotFoundError",
    "InvalidConfigurationError",
    "ProviderConnectionError",
]

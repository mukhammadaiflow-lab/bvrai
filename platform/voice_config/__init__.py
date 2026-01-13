"""
Voice Configuration Dashboard Module

This module provides a comprehensive voice configuration system that allows
users to configure STT (Speech-to-Text), TTS (Text-to-Speech), and voice
settings for AI agents. Better than competitors like Vapi with more
flexibility, more providers, and advanced features.

Key Features:
- Multi-provider STT support (9 providers)
- Multi-provider TTS support (11 providers)
- Voice library with 50+ pre-configured voices
- Custom voice ID support
- Model selection per provider
- Voice preview and testing
- Configuration presets (5 system presets)
- Provider health monitoring

Example usage:

    from platform.voice_config import (
        VoiceConfigurationService,
        STTProvider,
        TTSProvider,
        LanguageCode,
        VoiceGender,
        VoiceStyle,
    )

    # Initialize service
    service = VoiceConfigurationService()

    # Configure providers with API keys
    service.configure_stt_provider(
        provider=STTProvider.DEEPGRAM,
        api_key="your_deepgram_key",
    )

    service.configure_tts_provider(
        provider=TTSProvider.ELEVENLABS,
        api_key="your_elevenlabs_key",
    )

    # Browse available voices
    voices = service.get_available_voices(
        provider=TTSProvider.ELEVENLABS,
        language=LanguageCode.EN_US,
        gender=VoiceGender.FEMALE,
    )

    # Search voices
    voices = service.search_voices("friendly conversational")

    # Create a voice configuration
    config = await service.create_configuration(
        organization_id="org_123",
        agent_id="agent_456",
        name="Support Agent Voice",
        preset_id="preset_low_latency",  # Apply preset
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel from ElevenLabs
    )

    # Or create with specific settings
    config = await service.create_configuration(
        organization_id="org_123",
        agent_id="agent_789",
        name="Custom Voice Config",
        stt_provider=STTProvider.DEEPGRAM,
        stt_model_id="nova-2",
        tts_provider=TTSProvider.ELEVENLABS,
        tts_model_id="eleven_turbo_v2_5",
        voice_id="21m00Tcm4TlvDq8ikWAM",
        language=LanguageCode.EN_US,
        vad_enabled=True,
        allow_interruption=True,
        backchanneling_enabled=True,
        target_latency_ms=400,
    )

    # Add custom voice ID (manually specified)
    custom_voice = service.add_custom_voice(
        provider=TTSProvider.ELEVENLABS,
        voice_id="your_custom_voice_id",
        name="my_custom_voice",
        display_name="My Custom Voice",
    )

    # Update configuration
    await service.update_configuration(
        config_id=config.id,
        tts={"voice_id": custom_voice.voice_id},
    )

    # Get recommended voices for a use case
    recommended = service.get_recommended_voices(
        use_case="customer service",
        language=LanguageCode.EN_US,
    )

    # Get recommended presets
    presets = service.get_recommended_presets("conversational AI")

    # Generate voice preview
    audio = await service.generate_voice_preview(
        provider=TTSProvider.ELEVENLABS,
        voice_id="21m00Tcm4TlvDq8ikWAM",
        text="Hello, this is a preview of my voice.",
    )

    # Check provider health
    health = await service.check_provider_health()

    # Get service status
    status = await service.get_status()
    print(f"Available voices: {status['voice_library']['total_voices']}")

Provider Support:

    STT Providers (9):
    - Deepgram (recommended for low latency)
    - OpenAI Whisper (best multilingual)
    - Google Cloud Speech
    - Azure Speech Services
    - AssemblyAI
    - Amazon Transcribe
    - Speechmatics
    - Rev AI
    - Whisper (local)

    TTS Providers (11):
    - ElevenLabs (best quality, 13 voices)
    - OpenAI TTS (6 voices)
    - Azure Speech (6 voices)
    - Google Cloud TTS (4 voices)
    - PlayHT (2 voices)
    - Cartesia (ultra-low latency, 2 voices)
    - Deepgram Aura (5 voices)
    - Amazon Polly (4 voices)
    - Rime (1 voice)
    - WellSaid Labs
    - Murf AI

Configuration Presets:
    - Low Latency: Optimized for conversational AI (~300ms)
    - High Quality: Premium audio quality
    - Cost Optimized: Budget-friendly for high volume
    - Multilingual: Best multi-language support
    - Natural Conversation: Human-like with backchanneling
"""

# Base types and enums
from .base import (
    # Provider enums
    STTProvider,
    TTSProvider,
    # Voice enums
    VoiceGender,
    VoiceStyle,
    VoiceAge,
    LanguageCode,
    # STT types
    STTModelConfig,
    STTProviderConfig,
    STTConfiguration,
    # TTS types
    Voice,
    TTSModelConfig,
    TTSProviderConfig,
    VoiceSettings,
    TTSConfiguration,
    # Main configuration
    VoiceConfiguration,
    VoiceConfigurationPreset,
    # Exceptions
    VoiceConfigurationError,
    ProviderNotFoundError,
    VoiceNotFoundError,
    ModelNotFoundError,
    InvalidConfigurationError,
    ProviderConnectionError,
)

# Services
from .service import (
    # Managers
    STTProviderManager,
    TTSProviderManager,
    VoiceConfigurationManager,
    PresetManager,
    # Library
    VoiceLibrary,
    VoiceSearchFilters,
    # Services
    VoicePreviewService,
    ProviderHealthChecker,
    # Main Service
    VoiceConfigurationService,
    # Utility functions
    get_default_stt_providers,
    get_default_tts_providers,
    get_default_presets,
)


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
    # Managers
    "STTProviderManager",
    "TTSProviderManager",
    "VoiceConfigurationManager",
    "PresetManager",
    # Library
    "VoiceLibrary",
    "VoiceSearchFilters",
    # Services
    "VoicePreviewService",
    "ProviderHealthChecker",
    # Main Service
    "VoiceConfigurationService",
    # Utility functions
    "get_default_stt_providers",
    "get_default_tts_providers",
    "get_default_presets",
    # Exceptions
    "VoiceConfigurationError",
    "ProviderNotFoundError",
    "VoiceNotFoundError",
    "ModelNotFoundError",
    "InvalidConfigurationError",
    "ProviderConnectionError",
]

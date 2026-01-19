"""
Voice Configuration Service Module

This module provides the service layer for voice configuration management,
including provider management, voice library, presets, and configuration CRUD.

Key Services:
- VoiceConfigurationManager: CRUD for voice configurations
- STTProviderManager: Manage STT providers and models
- TTSProviderManager: Manage TTS providers, models, and voices
- VoiceLibrary: Browse and search voices across all providers
- PresetManager: Quick configuration presets
- ProviderHealthChecker: Monitor provider availability
- VoicePreviewService: Generate voice previews
- VoiceConfigurationService: Main orchestrating service
"""

import asyncio
import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from .base import (
    # Enums
    STTProvider,
    TTSProvider,
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


# =============================================================================
# Provider Registry - Built-in Provider Definitions
# =============================================================================


def get_default_stt_providers() -> Dict[STTProvider, STTProviderConfig]:
    """Get default STT provider configurations with their models."""
    return {
        STTProvider.DEEPGRAM: STTProviderConfig(
            provider=STTProvider.DEEPGRAM,
            display_name="Deepgram",
            models=[
                STTModelConfig(
                    model_id="nova-2",
                    display_name="Nova-2",
                    provider=STTProvider.DEEPGRAM,
                    languages=[LanguageCode.EN_US, LanguageCode.EN_GB, LanguageCode.ES_ES],
                    supports_streaming=True,
                    supports_diarization=True,
                    supports_punctuation=True,
                    supports_word_timestamps=True,
                    latency_ms_estimate=150,
                    accuracy_tier="premium",
                    description="Best accuracy with lowest latency",
                    recommended_for=["conversational AI", "phone calls", "real-time"],
                ),
                STTModelConfig(
                    model_id="nova-2-phonecall",
                    display_name="Nova-2 Phone Call",
                    provider=STTProvider.DEEPGRAM,
                    languages=[LanguageCode.EN_US],
                    supports_streaming=True,
                    supports_diarization=True,
                    latency_ms_estimate=150,
                    accuracy_tier="premium",
                    description="Optimized for phone call audio quality",
                    recommended_for=["phone calls", "call centers"],
                ),
                STTModelConfig(
                    model_id="nova-2-meeting",
                    display_name="Nova-2 Meeting",
                    provider=STTProvider.DEEPGRAM,
                    languages=[LanguageCode.EN_US],
                    supports_streaming=True,
                    supports_diarization=True,
                    latency_ms_estimate=180,
                    accuracy_tier="premium",
                    description="Optimized for meeting recordings",
                    recommended_for=["meetings", "conferences"],
                ),
                STTModelConfig(
                    model_id="enhanced",
                    display_name="Enhanced",
                    provider=STTProvider.DEEPGRAM,
                    languages=[LanguageCode.EN_US, LanguageCode.ES_ES],
                    supports_streaming=True,
                    latency_ms_estimate=200,
                    accuracy_tier="enhanced",
                    description="Good balance of accuracy and cost",
                ),
                STTModelConfig(
                    model_id="base",
                    display_name="Base",
                    provider=STTProvider.DEEPGRAM,
                    languages=[LanguageCode.EN_US],
                    supports_streaming=True,
                    latency_ms_estimate=250,
                    accuracy_tier="standard",
                    description="Most cost-effective option",
                ),
            ],
            default_model_id="nova-2",
        ),
        STTProvider.OPENAI: STTProviderConfig(
            provider=STTProvider.OPENAI,
            display_name="OpenAI Whisper",
            models=[
                STTModelConfig(
                    model_id="whisper-1",
                    display_name="Whisper-1",
                    provider=STTProvider.OPENAI,
                    languages=[
                        LanguageCode.EN_US, LanguageCode.ES_ES, LanguageCode.FR_FR,
                        LanguageCode.DE_DE, LanguageCode.JA_JP, LanguageCode.ZH_CN,
                    ],
                    supports_streaming=False,
                    supports_diarization=False,
                    latency_ms_estimate=800,
                    accuracy_tier="premium",
                    description="Best multilingual support",
                    recommended_for=["multilingual", "transcription"],
                ),
            ],
            default_model_id="whisper-1",
        ),
        STTProvider.GOOGLE: STTProviderConfig(
            provider=STTProvider.GOOGLE,
            display_name="Google Cloud Speech",
            models=[
                STTModelConfig(
                    model_id="latest_long",
                    display_name="Latest Long",
                    provider=STTProvider.GOOGLE,
                    languages=[LanguageCode.EN_US, LanguageCode.ES_ES, LanguageCode.FR_FR],
                    supports_streaming=True,
                    supports_diarization=True,
                    supports_word_timestamps=True,
                    latency_ms_estimate=200,
                    accuracy_tier="premium",
                    description="Best for longer audio",
                ),
                STTModelConfig(
                    model_id="latest_short",
                    display_name="Latest Short",
                    provider=STTProvider.GOOGLE,
                    languages=[LanguageCode.EN_US, LanguageCode.ES_ES],
                    supports_streaming=True,
                    latency_ms_estimate=150,
                    accuracy_tier="premium",
                    description="Optimized for short utterances",
                    recommended_for=["conversational AI", "commands"],
                ),
                STTModelConfig(
                    model_id="phone_call",
                    display_name="Phone Call",
                    provider=STTProvider.GOOGLE,
                    languages=[LanguageCode.EN_US],
                    supports_streaming=True,
                    supports_diarization=True,
                    latency_ms_estimate=180,
                    accuracy_tier="enhanced",
                    description="Optimized for phone audio",
                ),
            ],
            default_model_id="latest_short",
        ),
        STTProvider.AZURE: STTProviderConfig(
            provider=STTProvider.AZURE,
            display_name="Azure Speech Services",
            models=[
                STTModelConfig(
                    model_id="default",
                    display_name="Default",
                    provider=STTProvider.AZURE,
                    languages=[
                        LanguageCode.EN_US, LanguageCode.ES_ES, LanguageCode.FR_FR,
                        LanguageCode.DE_DE, LanguageCode.JA_JP, LanguageCode.ZH_CN,
                    ],
                    supports_streaming=True,
                    supports_diarization=True,
                    supports_word_timestamps=True,
                    latency_ms_estimate=180,
                    accuracy_tier="premium",
                    description="Microsoft's production STT",
                ),
                STTModelConfig(
                    model_id="conversation",
                    display_name="Conversation",
                    provider=STTProvider.AZURE,
                    languages=[LanguageCode.EN_US],
                    supports_streaming=True,
                    supports_diarization=True,
                    latency_ms_estimate=200,
                    accuracy_tier="enhanced",
                    description="Optimized for conversations",
                ),
            ],
            default_model_id="default",
        ),
        STTProvider.ASSEMBLYAI: STTProviderConfig(
            provider=STTProvider.ASSEMBLYAI,
            display_name="AssemblyAI",
            models=[
                STTModelConfig(
                    model_id="best",
                    display_name="Best",
                    provider=STTProvider.ASSEMBLYAI,
                    languages=[LanguageCode.EN_US, LanguageCode.ES_ES],
                    supports_streaming=True,
                    supports_diarization=True,
                    supports_word_timestamps=True,
                    latency_ms_estimate=200,
                    accuracy_tier="premium",
                    description="Highest accuracy model",
                ),
                STTModelConfig(
                    model_id="nano",
                    display_name="Nano",
                    provider=STTProvider.ASSEMBLYAI,
                    languages=[LanguageCode.EN_US],
                    supports_streaming=True,
                    latency_ms_estimate=120,
                    accuracy_tier="standard",
                    description="Fastest, lowest cost",
                    recommended_for=["real-time", "low latency"],
                ),
            ],
            default_model_id="best",
        ),
        STTProvider.AWS_TRANSCRIBE: STTProviderConfig(
            provider=STTProvider.AWS_TRANSCRIBE,
            display_name="Amazon Transcribe",
            models=[
                STTModelConfig(
                    model_id="default",
                    display_name="Default",
                    provider=STTProvider.AWS_TRANSCRIBE,
                    languages=[LanguageCode.EN_US, LanguageCode.ES_ES, LanguageCode.FR_FR],
                    supports_streaming=True,
                    supports_diarization=True,
                    latency_ms_estimate=250,
                    accuracy_tier="enhanced",
                    description="AWS standard transcription",
                ),
                STTModelConfig(
                    model_id="medical",
                    display_name="Medical",
                    provider=STTProvider.AWS_TRANSCRIBE,
                    languages=[LanguageCode.EN_US],
                    supports_streaming=True,
                    latency_ms_estimate=280,
                    accuracy_tier="premium",
                    description="Medical terminology optimized",
                    recommended_for=["healthcare", "medical"],
                ),
            ],
            default_model_id="default",
        ),
        STTProvider.SPEECHMATICS: STTProviderConfig(
            provider=STTProvider.SPEECHMATICS,
            display_name="Speechmatics",
            models=[
                STTModelConfig(
                    model_id="enhanced",
                    display_name="Enhanced",
                    provider=STTProvider.SPEECHMATICS,
                    languages=[LanguageCode.EN_US, LanguageCode.EN_GB, LanguageCode.ES_ES],
                    supports_streaming=True,
                    supports_diarization=True,
                    supports_word_timestamps=True,
                    latency_ms_estimate=200,
                    accuracy_tier="premium",
                    description="UK-based high accuracy",
                ),
            ],
            default_model_id="enhanced",
        ),
        STTProvider.REV_AI: STTProviderConfig(
            provider=STTProvider.REV_AI,
            display_name="Rev AI",
            models=[
                STTModelConfig(
                    model_id="machine",
                    display_name="Machine",
                    provider=STTProvider.REV_AI,
                    languages=[LanguageCode.EN_US],
                    supports_streaming=True,
                    supports_diarization=True,
                    latency_ms_estimate=220,
                    accuracy_tier="enhanced",
                    description="Automated transcription",
                ),
            ],
            default_model_id="machine",
        ),
    }


def get_default_tts_providers() -> Dict[TTSProvider, TTSProviderConfig]:
    """Get default TTS provider configurations with their models and voices."""
    return {
        TTSProvider.ELEVENLABS: TTSProviderConfig(
            provider=TTSProvider.ELEVENLABS,
            display_name="ElevenLabs",
            models=[
                TTSModelConfig(
                    model_id="eleven_turbo_v2_5",
                    display_name="Turbo v2.5",
                    provider=TTSProvider.ELEVENLABS,
                    quality_tier="turbo",
                    latency_tier="low",
                    supports_streaming=True,
                    supports_emotions=True,
                    latency_ms_estimate=150,
                    description="Fastest, lowest latency",
                    recommended_for=["conversational AI", "real-time"],
                ),
                TTSModelConfig(
                    model_id="eleven_multilingual_v2",
                    display_name="Multilingual v2",
                    provider=TTSProvider.ELEVENLABS,
                    quality_tier="premium",
                    latency_tier="standard",
                    supports_streaming=True,
                    supports_emotions=True,
                    latency_ms_estimate=300,
                    description="Best multilingual support",
                    recommended_for=["multilingual", "high quality"],
                ),
                TTSModelConfig(
                    model_id="eleven_monolingual_v1",
                    display_name="Monolingual v1",
                    provider=TTSProvider.ELEVENLABS,
                    quality_tier="standard",
                    latency_tier="standard",
                    supports_streaming=True,
                    latency_ms_estimate=250,
                    description="English optimized",
                ),
            ],
            voices=[
                Voice(
                    id="el_rachel",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="21m00Tcm4TlvDq8ikWAM",
                    name="rachel",
                    display_name="Rachel",
                    gender=VoiceGender.FEMALE,
                    age=VoiceAge.MIDDLE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    categories=["conversational", "assistant"],
                    tags=["warm", "natural", "american"],
                    description="Warm, conversational female voice",
                ),
                Voice(
                    id="el_drew",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="29vD33N1CtxCmqQRPOHJ",
                    name="drew",
                    display_name="Drew",
                    gender=VoiceGender.MALE,
                    age=VoiceAge.MIDDLE,
                    style=VoiceStyle.PROFESSIONAL,
                    language=LanguageCode.EN_US,
                    categories=["professional", "business"],
                    tags=["confident", "clear", "american"],
                    description="Professional male voice",
                ),
                Voice(
                    id="el_clyde",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="2EiwWnXFnvU5JabPnv8n",
                    name="clyde",
                    display_name="Clyde",
                    gender=VoiceGender.MALE,
                    age=VoiceAge.MATURE,
                    style=VoiceStyle.NARRATIVE,
                    language=LanguageCode.EN_US,
                    categories=["narrative", "storytelling"],
                    tags=["deep", "gravelly", "character"],
                    description="Deep, gravelly narrator voice",
                ),
                Voice(
                    id="el_domi",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="AZnzlk1XvdvUeBnXmlld",
                    name="domi",
                    display_name="Domi",
                    gender=VoiceGender.FEMALE,
                    age=VoiceAge.YOUNG,
                    style=VoiceStyle.ENERGETIC,
                    language=LanguageCode.EN_US,
                    categories=["energetic", "young"],
                    tags=["energetic", "upbeat", "american"],
                    description="Young, energetic female voice",
                ),
                Voice(
                    id="el_dave",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="CYw3kZ02Hs0563khs1Fj",
                    name="dave",
                    display_name="Dave",
                    gender=VoiceGender.MALE,
                    age=VoiceAge.YOUNG,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_GB,
                    accent="british",
                    categories=["conversational", "british"],
                    tags=["friendly", "british", "casual"],
                    description="Friendly British male voice",
                ),
                Voice(
                    id="el_fin",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="D38z5RcWu1voky8WS1ja",
                    name="fin",
                    display_name="Fin",
                    gender=VoiceGender.MALE,
                    age=VoiceAge.MIDDLE,
                    style=VoiceStyle.AUTHORITATIVE,
                    language=LanguageCode.EN_US,
                    categories=["authoritative", "news"],
                    tags=["authoritative", "news", "american"],
                    description="Authoritative news anchor voice",
                ),
                Voice(
                    id="el_sarah",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="EXAVITQu4vr4xnSDxMaL",
                    name="sarah",
                    display_name="Sarah",
                    gender=VoiceGender.FEMALE,
                    age=VoiceAge.YOUNG,
                    style=VoiceStyle.FRIENDLY,
                    language=LanguageCode.EN_US,
                    categories=["friendly", "assistant"],
                    tags=["soft", "gentle", "american"],
                    description="Soft, gentle female voice",
                ),
                Voice(
                    id="el_antoni",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="ErXwobaYiN019PkySvjV",
                    name="antoni",
                    display_name="Antoni",
                    gender=VoiceGender.MALE,
                    age=VoiceAge.MIDDLE,
                    style=VoiceStyle.WARM,
                    language=LanguageCode.EN_US,
                    categories=["warm", "conversational"],
                    tags=["warm", "friendly", "american"],
                    description="Warm, friendly male voice",
                ),
                Voice(
                    id="el_elli",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="MF3mGyEYCl7XYWbV9V6O",
                    name="elli",
                    display_name="Elli",
                    gender=VoiceGender.FEMALE,
                    age=VoiceAge.YOUNG,
                    style=VoiceStyle.FRIENDLY,
                    language=LanguageCode.EN_US,
                    categories=["friendly", "young"],
                    tags=["youthful", "bright", "american"],
                    description="Youthful, bright female voice",
                ),
                Voice(
                    id="el_josh",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="TxGEqnHWrfWFTfGW9XjX",
                    name="josh",
                    display_name="Josh",
                    gender=VoiceGender.MALE,
                    age=VoiceAge.YOUNG,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    categories=["conversational", "young"],
                    tags=["natural", "casual", "american"],
                    description="Natural, casual male voice",
                ),
                Voice(
                    id="el_arnold",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="VR6AewLTigWG4xSOukaG",
                    name="arnold",
                    display_name="Arnold",
                    gender=VoiceGender.MALE,
                    age=VoiceAge.MATURE,
                    style=VoiceStyle.AUTHORITATIVE,
                    language=LanguageCode.EN_US,
                    categories=["authoritative", "character"],
                    tags=["deep", "commanding", "american"],
                    description="Deep, commanding male voice",
                ),
                Voice(
                    id="el_adam",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="pNInz6obpgDQGcFmaJgB",
                    name="adam",
                    display_name="Adam",
                    gender=VoiceGender.MALE,
                    age=VoiceAge.MIDDLE,
                    style=VoiceStyle.NARRATIVE,
                    language=LanguageCode.EN_US,
                    categories=["narrative", "professional"],
                    tags=["deep", "clear", "american"],
                    description="Deep, clear narrator voice",
                ),
                Voice(
                    id="el_sam",
                    provider=TTSProvider.ELEVENLABS,
                    voice_id="yoZ06aMxZJJ28mfd3POQ",
                    name="sam",
                    display_name="Sam",
                    gender=VoiceGender.MALE,
                    age=VoiceAge.YOUNG,
                    style=VoiceStyle.ENERGETIC,
                    language=LanguageCode.EN_US,
                    categories=["energetic", "young"],
                    tags=["dynamic", "energetic", "american"],
                    description="Dynamic, energetic male voice",
                ),
            ],
            default_model_id="eleven_turbo_v2_5",
            default_voice_id="21m00Tcm4TlvDq8ikWAM",
        ),
        TTSProvider.OPENAI: TTSProviderConfig(
            provider=TTSProvider.OPENAI,
            display_name="OpenAI TTS",
            models=[
                TTSModelConfig(
                    model_id="tts-1",
                    display_name="TTS-1",
                    provider=TTSProvider.OPENAI,
                    quality_tier="standard",
                    latency_tier="low",
                    supports_streaming=True,
                    latency_ms_estimate=200,
                    description="Fast, good quality",
                    recommended_for=["real-time", "conversational"],
                ),
                TTSModelConfig(
                    model_id="tts-1-hd",
                    display_name="TTS-1 HD",
                    provider=TTSProvider.OPENAI,
                    quality_tier="premium",
                    latency_tier="standard",
                    supports_streaming=True,
                    latency_ms_estimate=400,
                    description="Highest quality",
                    recommended_for=["high quality", "production"],
                ),
            ],
            voices=[
                Voice(
                    id="oai_alloy",
                    provider=TTSProvider.OPENAI,
                    voice_id="alloy",
                    name="alloy",
                    display_name="Alloy",
                    gender=VoiceGender.NEUTRAL,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    categories=["neutral", "assistant"],
                    tags=["neutral", "balanced"],
                    description="Balanced, neutral voice",
                ),
                Voice(
                    id="oai_echo",
                    provider=TTSProvider.OPENAI,
                    voice_id="echo",
                    name="echo",
                    display_name="Echo",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    categories=["conversational", "male"],
                    tags=["smooth", "calm"],
                    description="Smooth male voice",
                ),
                Voice(
                    id="oai_fable",
                    provider=TTSProvider.OPENAI,
                    voice_id="fable",
                    name="fable",
                    display_name="Fable",
                    gender=VoiceGender.NEUTRAL,
                    style=VoiceStyle.NARRATIVE,
                    language=LanguageCode.EN_GB,
                    accent="british",
                    categories=["narrative", "british"],
                    tags=["british", "storytelling"],
                    description="British narrative voice",
                ),
                Voice(
                    id="oai_onyx",
                    provider=TTSProvider.OPENAI,
                    voice_id="onyx",
                    name="onyx",
                    display_name="Onyx",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.AUTHORITATIVE,
                    language=LanguageCode.EN_US,
                    categories=["authoritative", "deep"],
                    tags=["deep", "authoritative"],
                    description="Deep, authoritative voice",
                ),
                Voice(
                    id="oai_nova",
                    provider=TTSProvider.OPENAI,
                    voice_id="nova",
                    name="nova",
                    display_name="Nova",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.FRIENDLY,
                    language=LanguageCode.EN_US,
                    categories=["friendly", "female"],
                    tags=["friendly", "warm"],
                    description="Warm, friendly female voice",
                ),
                Voice(
                    id="oai_shimmer",
                    provider=TTSProvider.OPENAI,
                    voice_id="shimmer",
                    name="shimmer",
                    display_name="Shimmer",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.CALM,
                    language=LanguageCode.EN_US,
                    categories=["calm", "female"],
                    tags=["gentle", "soft"],
                    description="Gentle, soft female voice",
                ),
            ],
            default_model_id="tts-1",
            default_voice_id="alloy",
        ),
        TTSProvider.AZURE: TTSProviderConfig(
            provider=TTSProvider.AZURE,
            display_name="Azure Speech",
            models=[
                TTSModelConfig(
                    model_id="neural",
                    display_name="Neural",
                    provider=TTSProvider.AZURE,
                    quality_tier="premium",
                    latency_tier="standard",
                    supports_ssml=True,
                    supports_streaming=True,
                    supports_emotions=True,
                    latency_ms_estimate=250,
                    description="Neural TTS with emotion support",
                ),
            ],
            voices=[
                Voice(
                    id="az_jenny",
                    provider=TTSProvider.AZURE,
                    voice_id="en-US-JennyNeural",
                    name="jenny",
                    display_name="Jenny (US)",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    supports_ssml=True,
                    categories=["conversational", "assistant"],
                    tags=["neural", "american"],
                    description="American female neural voice",
                ),
                Voice(
                    id="az_guy",
                    provider=TTSProvider.AZURE,
                    voice_id="en-US-GuyNeural",
                    name="guy",
                    display_name="Guy (US)",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.PROFESSIONAL,
                    language=LanguageCode.EN_US,
                    supports_ssml=True,
                    categories=["professional", "news"],
                    tags=["neural", "american"],
                    description="American male neural voice",
                ),
                Voice(
                    id="az_aria",
                    provider=TTSProvider.AZURE,
                    voice_id="en-US-AriaNeural",
                    name="aria",
                    display_name="Aria (US)",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.FRIENDLY,
                    language=LanguageCode.EN_US,
                    supports_ssml=True,
                    categories=["friendly", "assistant"],
                    tags=["neural", "expressive", "american"],
                    description="Expressive American female voice",
                ),
                Voice(
                    id="az_davis",
                    provider=TTSProvider.AZURE,
                    voice_id="en-US-DavisNeural",
                    name="davis",
                    display_name="Davis (US)",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    supports_ssml=True,
                    categories=["conversational", "casual"],
                    tags=["neural", "casual", "american"],
                    description="Casual American male voice",
                ),
                Voice(
                    id="az_sonia",
                    provider=TTSProvider.AZURE,
                    voice_id="en-GB-SoniaNeural",
                    name="sonia",
                    display_name="Sonia (UK)",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.PROFESSIONAL,
                    language=LanguageCode.EN_GB,
                    accent="british",
                    supports_ssml=True,
                    categories=["professional", "british"],
                    tags=["neural", "british"],
                    description="British female neural voice",
                ),
                Voice(
                    id="az_ryan",
                    provider=TTSProvider.AZURE,
                    voice_id="en-GB-RyanNeural",
                    name="ryan",
                    display_name="Ryan (UK)",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_GB,
                    accent="british",
                    supports_ssml=True,
                    categories=["conversational", "british"],
                    tags=["neural", "british"],
                    description="British male neural voice",
                ),
            ],
            default_model_id="neural",
            default_voice_id="en-US-JennyNeural",
        ),
        TTSProvider.GOOGLE: TTSProviderConfig(
            provider=TTSProvider.GOOGLE,
            display_name="Google Cloud TTS",
            models=[
                TTSModelConfig(
                    model_id="standard",
                    display_name="Standard",
                    provider=TTSProvider.GOOGLE,
                    quality_tier="standard",
                    latency_tier="low",
                    supports_ssml=True,
                    latency_ms_estimate=150,
                    description="Standard quality, low latency",
                ),
                TTSModelConfig(
                    model_id="wavenet",
                    display_name="WaveNet",
                    provider=TTSProvider.GOOGLE,
                    quality_tier="premium",
                    latency_tier="standard",
                    supports_ssml=True,
                    latency_ms_estimate=300,
                    description="High quality WaveNet voices",
                    recommended_for=["high quality", "natural"],
                ),
                TTSModelConfig(
                    model_id="neural2",
                    display_name="Neural2",
                    provider=TTSProvider.GOOGLE,
                    quality_tier="premium",
                    latency_tier="standard",
                    supports_ssml=True,
                    latency_ms_estimate=280,
                    description="Latest neural voices",
                    recommended_for=["conversational", "expressive"],
                ),
            ],
            voices=[
                Voice(
                    id="gc_wavenet_a",
                    provider=TTSProvider.GOOGLE,
                    voice_id="en-US-Wavenet-A",
                    name="wavenet_a",
                    display_name="WaveNet A (US Male)",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.PROFESSIONAL,
                    language=LanguageCode.EN_US,
                    categories=["professional", "wavenet"],
                    tags=["wavenet", "american"],
                    description="American male WaveNet voice",
                ),
                Voice(
                    id="gc_wavenet_c",
                    provider=TTSProvider.GOOGLE,
                    voice_id="en-US-Wavenet-C",
                    name="wavenet_c",
                    display_name="WaveNet C (US Female)",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.FRIENDLY,
                    language=LanguageCode.EN_US,
                    categories=["friendly", "wavenet"],
                    tags=["wavenet", "american"],
                    description="American female WaveNet voice",
                ),
                Voice(
                    id="gc_neural2_a",
                    provider=TTSProvider.GOOGLE,
                    voice_id="en-US-Neural2-A",
                    name="neural2_a",
                    display_name="Neural2 A (US Male)",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    categories=["conversational", "neural"],
                    tags=["neural2", "american"],
                    description="American male Neural2 voice",
                ),
                Voice(
                    id="gc_neural2_c",
                    provider=TTSProvider.GOOGLE,
                    voice_id="en-US-Neural2-C",
                    name="neural2_c",
                    display_name="Neural2 C (US Female)",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    categories=["conversational", "neural"],
                    tags=["neural2", "american"],
                    description="American female Neural2 voice",
                ),
            ],
            default_model_id="neural2",
            default_voice_id="en-US-Neural2-C",
        ),
        TTSProvider.PLAYHT: TTSProviderConfig(
            provider=TTSProvider.PLAYHT,
            display_name="PlayHT",
            models=[
                TTSModelConfig(
                    model_id="PlayHT2.0-turbo",
                    display_name="PlayHT 2.0 Turbo",
                    provider=TTSProvider.PLAYHT,
                    quality_tier="turbo",
                    latency_tier="low",
                    supports_streaming=True,
                    supports_emotions=True,
                    latency_ms_estimate=180,
                    description="Fastest, lowest latency",
                    recommended_for=["conversational AI", "real-time"],
                ),
                TTSModelConfig(
                    model_id="PlayHT2.0",
                    display_name="PlayHT 2.0",
                    provider=TTSProvider.PLAYHT,
                    quality_tier="premium",
                    latency_tier="standard",
                    supports_streaming=True,
                    supports_emotions=True,
                    latency_ms_estimate=350,
                    description="Highest quality",
                    recommended_for=["high quality", "production"],
                ),
            ],
            voices=[
                Voice(
                    id="ph_jennifer",
                    provider=TTSProvider.PLAYHT,
                    voice_id="s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
                    name="jennifer",
                    display_name="Jennifer",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.CUSTOMER_SERVICE,
                    language=LanguageCode.EN_US,
                    categories=["customer_service", "professional"],
                    tags=["professional", "american"],
                    description="Professional customer service voice",
                ),
                Voice(
                    id="ph_jack",
                    provider=TTSProvider.PLAYHT,
                    voice_id="s3://voice-cloning-zero-shot/801a663f-efd0-4254-98d0-5c175514c3e8/male-2/manifest.json",
                    name="jack",
                    display_name="Jack",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    categories=["conversational", "friendly"],
                    tags=["friendly", "american"],
                    description="Friendly conversational male voice",
                ),
            ],
            default_model_id="PlayHT2.0-turbo",
            default_voice_id="s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
        ),
        TTSProvider.CARTESIA: TTSProviderConfig(
            provider=TTSProvider.CARTESIA,
            display_name="Cartesia",
            models=[
                TTSModelConfig(
                    model_id="sonic-english",
                    display_name="Sonic English",
                    provider=TTSProvider.CARTESIA,
                    quality_tier="turbo",
                    latency_tier="low",
                    supports_streaming=True,
                    latency_ms_estimate=100,
                    description="Ultra-low latency, English optimized",
                    recommended_for=["real-time", "conversational AI", "low latency"],
                ),
                TTSModelConfig(
                    model_id="sonic-multilingual",
                    display_name="Sonic Multilingual",
                    provider=TTSProvider.CARTESIA,
                    quality_tier="turbo",
                    latency_tier="low",
                    supports_streaming=True,
                    latency_ms_estimate=120,
                    description="Ultra-low latency, multilingual",
                    recommended_for=["multilingual", "real-time"],
                ),
            ],
            voices=[
                Voice(
                    id="cart_confident",
                    provider=TTSProvider.CARTESIA,
                    voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
                    name="confident_male",
                    display_name="Confident Male",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.PROFESSIONAL,
                    language=LanguageCode.EN_US,
                    categories=["professional", "business"],
                    tags=["confident", "american"],
                    description="Confident professional male voice",
                ),
                Voice(
                    id="cart_friendly",
                    provider=TTSProvider.CARTESIA,
                    voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
                    name="friendly_female",
                    display_name="Friendly Female",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.FRIENDLY,
                    language=LanguageCode.EN_US,
                    categories=["friendly", "assistant"],
                    tags=["friendly", "warm", "american"],
                    description="Warm, friendly female voice",
                ),
            ],
            default_model_id="sonic-english",
            default_voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
        ),
        TTSProvider.DEEPGRAM: TTSProviderConfig(
            provider=TTSProvider.DEEPGRAM,
            display_name="Deepgram TTS",
            models=[
                TTSModelConfig(
                    model_id="aura",
                    display_name="Aura",
                    provider=TTSProvider.DEEPGRAM,
                    quality_tier="premium",
                    latency_tier="low",
                    supports_streaming=True,
                    latency_ms_estimate=150,
                    description="Fast, natural-sounding voices",
                    recommended_for=["conversational AI", "real-time"],
                ),
            ],
            voices=[
                Voice(
                    id="dg_asteria",
                    provider=TTSProvider.DEEPGRAM,
                    voice_id="aura-asteria-en",
                    name="asteria",
                    display_name="Asteria",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    categories=["conversational", "assistant"],
                    tags=["natural", "american"],
                    description="Natural conversational female voice",
                ),
                Voice(
                    id="dg_luna",
                    provider=TTSProvider.DEEPGRAM,
                    voice_id="aura-luna-en",
                    name="luna",
                    display_name="Luna",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.WARM,
                    language=LanguageCode.EN_US,
                    categories=["warm", "friendly"],
                    tags=["warm", "soft", "american"],
                    description="Warm, soft female voice",
                ),
                Voice(
                    id="dg_stella",
                    provider=TTSProvider.DEEPGRAM,
                    voice_id="aura-stella-en",
                    name="stella",
                    display_name="Stella",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.PROFESSIONAL,
                    language=LanguageCode.EN_US,
                    categories=["professional", "business"],
                    tags=["professional", "clear", "american"],
                    description="Professional female voice",
                ),
                Voice(
                    id="dg_orion",
                    provider=TTSProvider.DEEPGRAM,
                    voice_id="aura-orion-en",
                    name="orion",
                    display_name="Orion",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.PROFESSIONAL,
                    language=LanguageCode.EN_US,
                    categories=["professional", "authoritative"],
                    tags=["authoritative", "clear", "american"],
                    description="Authoritative male voice",
                ),
                Voice(
                    id="dg_arcas",
                    provider=TTSProvider.DEEPGRAM,
                    voice_id="aura-arcas-en",
                    name="arcas",
                    display_name="Arcas",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    categories=["conversational", "friendly"],
                    tags=["friendly", "casual", "american"],
                    description="Friendly conversational male voice",
                ),
            ],
            default_model_id="aura",
            default_voice_id="aura-asteria-en",
        ),
        TTSProvider.AWS_POLLY: TTSProviderConfig(
            provider=TTSProvider.AWS_POLLY,
            display_name="Amazon Polly",
            models=[
                TTSModelConfig(
                    model_id="standard",
                    display_name="Standard",
                    provider=TTSProvider.AWS_POLLY,
                    quality_tier="standard",
                    latency_tier="low",
                    supports_ssml=True,
                    latency_ms_estimate=150,
                    description="Standard quality voices",
                ),
                TTSModelConfig(
                    model_id="neural",
                    display_name="Neural",
                    provider=TTSProvider.AWS_POLLY,
                    quality_tier="premium",
                    latency_tier="standard",
                    supports_ssml=True,
                    latency_ms_estimate=250,
                    description="Neural TTS voices",
                    recommended_for=["high quality", "natural"],
                ),
            ],
            voices=[
                Voice(
                    id="polly_joanna",
                    provider=TTSProvider.AWS_POLLY,
                    voice_id="Joanna",
                    name="joanna",
                    display_name="Joanna (US)",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    categories=["conversational", "assistant"],
                    tags=["neural", "american"],
                    description="American female neural voice",
                ),
                Voice(
                    id="polly_matthew",
                    provider=TTSProvider.AWS_POLLY,
                    voice_id="Matthew",
                    name="matthew",
                    display_name="Matthew (US)",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    categories=["conversational", "professional"],
                    tags=["neural", "american"],
                    description="American male neural voice",
                ),
                Voice(
                    id="polly_amy",
                    provider=TTSProvider.AWS_POLLY,
                    voice_id="Amy",
                    name="amy",
                    display_name="Amy (UK)",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.PROFESSIONAL,
                    language=LanguageCode.EN_GB,
                    accent="british",
                    categories=["professional", "british"],
                    tags=["neural", "british"],
                    description="British female neural voice",
                ),
                Voice(
                    id="polly_brian",
                    provider=TTSProvider.AWS_POLLY,
                    voice_id="Brian",
                    name="brian",
                    display_name="Brian (UK)",
                    gender=VoiceGender.MALE,
                    style=VoiceStyle.PROFESSIONAL,
                    language=LanguageCode.EN_GB,
                    accent="british",
                    categories=["professional", "british"],
                    tags=["neural", "british"],
                    description="British male neural voice",
                ),
            ],
            default_model_id="neural",
            default_voice_id="Joanna",
        ),
        TTSProvider.RIME: TTSProviderConfig(
            provider=TTSProvider.RIME,
            display_name="Rime",
            models=[
                TTSModelConfig(
                    model_id="mist",
                    display_name="Mist",
                    provider=TTSProvider.RIME,
                    quality_tier="turbo",
                    latency_tier="low",
                    supports_streaming=True,
                    latency_ms_estimate=100,
                    description="Ultra-low latency model",
                    recommended_for=["real-time", "conversational AI"],
                ),
            ],
            voices=[
                Voice(
                    id="rime_luna",
                    provider=TTSProvider.RIME,
                    voice_id="luna",
                    name="luna",
                    display_name="Luna",
                    gender=VoiceGender.FEMALE,
                    style=VoiceStyle.CONVERSATIONAL,
                    language=LanguageCode.EN_US,
                    categories=["conversational", "assistant"],
                    tags=["natural", "american"],
                    description="Natural conversational voice",
                ),
            ],
            default_model_id="mist",
            default_voice_id="luna",
        ),
    }


def get_default_presets() -> List[VoiceConfigurationPreset]:
    """Get default configuration presets."""
    return [
        VoiceConfigurationPreset(
            id="preset_low_latency",
            name="Low Latency",
            description="Optimized for conversational AI with minimal delay",
            category="conversational",
            stt_provider=STTProvider.DEEPGRAM,
            stt_model_id="nova-2",
            tts_provider=TTSProvider.CARTESIA,
            tts_model_id="sonic-english",
            voice_settings=VoiceSettings(speed=1.0, stability=0.5),
            vad_enabled=True,
            turn_detection_enabled=True,
            allow_interruption=True,
            backchanneling_enabled=False,
            filler_enabled=False,
            noise_suppression=True,
            target_latency_ms=300,
            estimated_cost_per_minute=0.08,
            recommended_for=["phone calls", "customer service", "conversational AI"],
            is_system=True,
        ),
        VoiceConfigurationPreset(
            id="preset_high_quality",
            name="High Quality",
            description="Best audio quality for premium experiences",
            category="high_quality",
            stt_provider=STTProvider.DEEPGRAM,
            stt_model_id="nova-2",
            tts_provider=TTSProvider.ELEVENLABS,
            tts_model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                speed=1.0,
                stability=0.7,
                similarity_boost=0.8,
                style=0.3,
            ),
            vad_enabled=True,
            turn_detection_enabled=True,
            allow_interruption=True,
            backchanneling_enabled=False,
            filler_enabled=False,
            noise_suppression=True,
            target_latency_ms=600,
            estimated_cost_per_minute=0.15,
            recommended_for=["premium support", "VIP customers", "demos"],
            is_system=True,
        ),
        VoiceConfigurationPreset(
            id="preset_cost_optimized",
            name="Cost Optimized",
            description="Balanced quality at lower cost for high volume",
            category="cost_optimized",
            stt_provider=STTProvider.DEEPGRAM,
            stt_model_id="enhanced",
            tts_provider=TTSProvider.OPENAI,
            tts_model_id="tts-1",
            voice_settings=VoiceSettings(speed=1.0),
            vad_enabled=True,
            turn_detection_enabled=True,
            allow_interruption=True,
            noise_suppression=True,
            target_latency_ms=500,
            estimated_cost_per_minute=0.05,
            recommended_for=["high volume", "cost-sensitive", "batch processing"],
            is_system=True,
        ),
        VoiceConfigurationPreset(
            id="preset_multilingual",
            name="Multilingual",
            description="Best support for multiple languages",
            category="multilingual",
            stt_provider=STTProvider.OPENAI,
            stt_model_id="whisper-1",
            tts_provider=TTSProvider.ELEVENLABS,
            tts_model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(speed=1.0, stability=0.6),
            vad_enabled=True,
            turn_detection_enabled=True,
            allow_interruption=True,
            noise_suppression=True,
            target_latency_ms=800,
            estimated_cost_per_minute=0.12,
            recommended_for=["international", "multilingual support"],
            is_system=True,
        ),
        VoiceConfigurationPreset(
            id="preset_natural_conversation",
            name="Natural Conversation",
            description="Human-like conversation with backchanneling",
            category="conversational",
            stt_provider=STTProvider.DEEPGRAM,
            stt_model_id="nova-2",
            tts_provider=TTSProvider.ELEVENLABS,
            tts_model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                speed=1.0,
                stability=0.4,
                similarity_boost=0.75,
            ),
            vad_enabled=True,
            turn_detection_enabled=True,
            allow_interruption=True,
            backchanneling_enabled=True,
            filler_enabled=True,
            noise_suppression=True,
            target_latency_ms=400,
            estimated_cost_per_minute=0.10,
            recommended_for=["human-like AI", "therapy", "coaching"],
            is_system=True,
        ),
    ]


# =============================================================================
# Provider Manager Classes
# =============================================================================


class STTProviderManager:
    """Manager for STT providers."""

    def __init__(self):
        self._providers: Dict[STTProvider, STTProviderConfig] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load default provider configurations."""
        self._providers = get_default_stt_providers()

    def get_provider(self, provider: STTProvider) -> Optional[STTProviderConfig]:
        """Get provider configuration."""
        return self._providers.get(provider)

    def get_all_providers(self) -> List[STTProviderConfig]:
        """Get all provider configurations."""
        return list(self._providers.values())

    def get_enabled_providers(self) -> List[STTProviderConfig]:
        """Get only enabled providers."""
        return [p for p in self._providers.values() if p.is_enabled]

    def configure_provider(
        self,
        provider: STTProvider,
        api_key: str,
        api_endpoint: Optional[str] = None,
        region: Optional[str] = None,
    ) -> STTProviderConfig:
        """Configure a provider with API credentials."""
        if provider not in self._providers:
            raise ProviderNotFoundError(f"STT provider not found: {provider}")

        config = self._providers[provider]
        config.api_key = api_key
        if api_endpoint:
            config.api_endpoint = api_endpoint
        if region:
            config.region = region
        config.is_enabled = True

        return config

    def enable_provider(self, provider: STTProvider) -> None:
        """Enable a provider."""
        if provider in self._providers:
            self._providers[provider].is_enabled = True

    def disable_provider(self, provider: STTProvider) -> None:
        """Disable a provider."""
        if provider in self._providers:
            self._providers[provider].is_enabled = False

    def get_models(self, provider: STTProvider) -> List[STTModelConfig]:
        """Get available models for a provider."""
        config = self._providers.get(provider)
        return config.models if config else []

    def get_model(
        self, provider: STTProvider, model_id: str
    ) -> Optional[STTModelConfig]:
        """Get a specific model configuration."""
        models = self.get_models(provider)
        return next((m for m in models if m.model_id == model_id), None)

    def add_custom_provider(self, config: STTProviderConfig) -> None:
        """Add a custom provider configuration."""
        self._providers[config.provider] = config

    def update_health(
        self,
        provider: STTProvider,
        is_healthy: bool,
        error_rate: float = 0.0,
    ) -> None:
        """Update provider health status."""
        if provider in self._providers:
            self._providers[provider].is_healthy = is_healthy
            self._providers[provider].error_rate = error_rate
            self._providers[provider].last_health_check = datetime.utcnow()

    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convert all providers to dictionary."""
        return {
            p.value: config.to_dict(include_secrets=include_secrets)
            for p, config in self._providers.items()
        }


class TTSProviderManager:
    """Manager for TTS providers and voices."""

    def __init__(self):
        self._providers: Dict[TTSProvider, TTSProviderConfig] = {}
        self._custom_voices: Dict[str, Voice] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load default provider configurations."""
        self._providers = get_default_tts_providers()

    def get_provider(self, provider: TTSProvider) -> Optional[TTSProviderConfig]:
        """Get provider configuration."""
        return self._providers.get(provider)

    def get_all_providers(self) -> List[TTSProviderConfig]:
        """Get all provider configurations."""
        return list(self._providers.values())

    def get_enabled_providers(self) -> List[TTSProviderConfig]:
        """Get only enabled providers."""
        return [p for p in self._providers.values() if p.is_enabled]

    def configure_provider(
        self,
        provider: TTSProvider,
        api_key: str,
        api_endpoint: Optional[str] = None,
        region: Optional[str] = None,
    ) -> TTSProviderConfig:
        """Configure a provider with API credentials."""
        if provider not in self._providers:
            raise ProviderNotFoundError(f"TTS provider not found: {provider}")

        config = self._providers[provider]
        config.api_key = api_key
        if api_endpoint:
            config.api_endpoint = api_endpoint
        if region:
            config.region = region
        config.is_enabled = True

        return config

    def enable_provider(self, provider: TTSProvider) -> None:
        """Enable a provider."""
        if provider in self._providers:
            self._providers[provider].is_enabled = True

    def disable_provider(self, provider: TTSProvider) -> None:
        """Disable a provider."""
        if provider in self._providers:
            self._providers[provider].is_enabled = False

    def get_models(self, provider: TTSProvider) -> List[TTSModelConfig]:
        """Get available models for a provider."""
        config = self._providers.get(provider)
        return config.models if config else []

    def get_model(
        self, provider: TTSProvider, model_id: str
    ) -> Optional[TTSModelConfig]:
        """Get a specific model configuration."""
        models = self.get_models(provider)
        return next((m for m in models if m.model_id == model_id), None)

    def get_voices(self, provider: TTSProvider) -> List[Voice]:
        """Get available voices for a provider."""
        config = self._providers.get(provider)
        voices = config.voices if config else []
        # Include custom voices for this provider
        custom = [v for v in self._custom_voices.values() if v.provider == provider]
        return voices + custom

    def get_voice(self, provider: TTSProvider, voice_id: str) -> Optional[Voice]:
        """Get a specific voice."""
        voices = self.get_voices(provider)
        return next((v for v in voices if v.voice_id == voice_id), None)

    def get_all_voices(self) -> List[Voice]:
        """Get all available voices across all providers."""
        all_voices = []
        for config in self._providers.values():
            all_voices.extend(config.voices)
        all_voices.extend(self._custom_voices.values())
        return all_voices

    def add_custom_voice(
        self,
        provider: TTSProvider,
        voice_id: str,
        name: str,
        display_name: str,
        gender: VoiceGender = VoiceGender.NEUTRAL,
        language: LanguageCode = LanguageCode.EN_US,
        description: str = "",
    ) -> Voice:
        """Add a custom voice ID (manually specified by user)."""
        voice = Voice(
            id=f"custom_{uuid.uuid4().hex[:12]}",
            provider=provider,
            voice_id=voice_id,
            name=name,
            display_name=display_name,
            gender=gender,
            language=language,
            is_custom=True,
            description=description or f"Custom voice: {voice_id}",
        )
        self._custom_voices[voice.id] = voice
        return voice

    def remove_custom_voice(self, voice_id: str) -> bool:
        """Remove a custom voice."""
        if voice_id in self._custom_voices:
            del self._custom_voices[voice_id]
            return True
        return False

    def get_custom_voices(self) -> List[Voice]:
        """Get all custom voices."""
        return list(self._custom_voices.values())

    def update_health(
        self,
        provider: TTSProvider,
        is_healthy: bool,
        error_rate: float = 0.0,
    ) -> None:
        """Update provider health status."""
        if provider in self._providers:
            self._providers[provider].is_healthy = is_healthy
            self._providers[provider].error_rate = error_rate
            self._providers[provider].last_health_check = datetime.utcnow()

    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convert all providers to dictionary."""
        return {
            p.value: config.to_dict(include_secrets=include_secrets)
            for p, config in self._providers.items()
        }


# =============================================================================
# Voice Library
# =============================================================================


@dataclass
class VoiceSearchFilters:
    """Filters for voice search."""

    providers: Optional[List[TTSProvider]] = None
    genders: Optional[List[VoiceGender]] = None
    languages: Optional[List[LanguageCode]] = None
    styles: Optional[List[VoiceStyle]] = None
    ages: Optional[List[VoiceAge]] = None
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    is_premium: Optional[bool] = None
    supports_streaming: Optional[bool] = None
    query: Optional[str] = None  # Text search in name/description


class VoiceLibrary:
    """
    Voice library for browsing and searching voices across all providers.

    This provides a unified interface to discover voices from all configured
    TTS providers, with filtering and search capabilities.
    """

    def __init__(self, tts_manager: TTSProviderManager):
        self._tts_manager = tts_manager
        self._favorites: Set[str] = set()
        self._recently_used: List[str] = []

    def get_all_voices(self) -> List[Voice]:
        """Get all available voices."""
        return self._tts_manager.get_all_voices()

    def search(
        self,
        filters: Optional[VoiceSearchFilters] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Voice], int]:
        """
        Search voices with filters.

        Returns (voices, total_count).
        """
        voices = self.get_all_voices()

        if filters:
            voices = self._apply_filters(voices, filters)

        total = len(voices)
        voices = voices[offset : offset + limit]

        return voices, total

    def _apply_filters(
        self, voices: List[Voice], filters: VoiceSearchFilters
    ) -> List[Voice]:
        """Apply filters to voice list."""
        result = voices

        if filters.providers:
            result = [v for v in result if v.provider in filters.providers]

        if filters.genders:
            result = [v for v in result if v.gender in filters.genders]

        if filters.languages:
            result = [v for v in result if v.language in filters.languages]

        if filters.styles:
            result = [v for v in result if v.style in filters.styles]

        if filters.ages:
            result = [v for v in result if v.age in filters.ages]

        if filters.categories:
            result = [
                v for v in result
                if any(c in v.categories for c in filters.categories)
            ]

        if filters.tags:
            result = [
                v for v in result
                if any(t in v.tags for t in filters.tags)
            ]

        if filters.is_premium is not None:
            result = [v for v in result if v.is_premium == filters.is_premium]

        if filters.supports_streaming is not None:
            result = [
                v for v in result
                if v.supports_streaming == filters.supports_streaming
            ]

        if filters.query:
            query = filters.query.lower()
            result = [
                v for v in result
                if query in v.name.lower()
                or query in v.display_name.lower()
                or query in v.description.lower()
                or any(query in t.lower() for t in v.tags)
            ]

        return result

    def get_by_provider(self, provider: TTSProvider) -> List[Voice]:
        """Get all voices for a specific provider."""
        return self._tts_manager.get_voices(provider)

    def get_by_language(self, language: LanguageCode) -> List[Voice]:
        """Get all voices supporting a specific language."""
        return [v for v in self.get_all_voices() if v.language == language]

    def get_by_gender(self, gender: VoiceGender) -> List[Voice]:
        """Get all voices of a specific gender."""
        return [v for v in self.get_all_voices() if v.gender == gender]

    def get_by_style(self, style: VoiceStyle) -> List[Voice]:
        """Get all voices with a specific style."""
        return [v for v in self.get_all_voices() if v.style == style]

    def get_by_category(self, category: str) -> List[Voice]:
        """Get all voices in a specific category."""
        return [v for v in self.get_all_voices() if category in v.categories]

    def get_recommended(
        self,
        use_case: str,
        language: LanguageCode = LanguageCode.EN_US,
        limit: int = 5,
    ) -> List[Voice]:
        """Get recommended voices for a specific use case."""
        voices = self.get_by_language(language)

        # Score voices based on use case matching
        scored = []
        use_case_lower = use_case.lower()

        for voice in voices:
            score = 0
            # Check categories
            if any(use_case_lower in c.lower() for c in voice.categories):
                score += 3
            # Check tags
            if any(use_case_lower in t.lower() for t in voice.tags):
                score += 2
            # Check style
            if use_case_lower in voice.style.value.lower():
                score += 2
            # Check description
            if use_case_lower in voice.description.lower():
                score += 1

            if score > 0:
                scored.append((voice, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [v for v, _ in scored[:limit]]

    def add_favorite(self, voice_id: str) -> None:
        """Add a voice to favorites."""
        self._favorites.add(voice_id)

    def remove_favorite(self, voice_id: str) -> None:
        """Remove a voice from favorites."""
        self._favorites.discard(voice_id)

    def get_favorites(self) -> List[Voice]:
        """Get favorite voices."""
        all_voices = {v.id: v for v in self.get_all_voices()}
        return [all_voices[vid] for vid in self._favorites if vid in all_voices]

    def record_usage(self, voice_id: str) -> None:
        """Record voice usage for recently used tracking."""
        if voice_id in self._recently_used:
            self._recently_used.remove(voice_id)
        self._recently_used.insert(0, voice_id)
        # Keep only last 20
        self._recently_used = self._recently_used[:20]

    def get_recently_used(self, limit: int = 10) -> List[Voice]:
        """Get recently used voices."""
        all_voices = {v.id: v for v in self.get_all_voices()}
        return [
            all_voices[vid]
            for vid in self._recently_used[:limit]
            if vid in all_voices
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get voice library statistics."""
        voices = self.get_all_voices()

        return {
            "total_voices": len(voices),
            "by_provider": {
                p.value: len(self.get_by_provider(p))
                for p in TTSProvider
            },
            "by_gender": {
                g.value: len(self.get_by_gender(g))
                for g in VoiceGender
            },
            "by_language": {
                l.value: len(self.get_by_language(l))
                for l in [LanguageCode.EN_US, LanguageCode.ES_ES, LanguageCode.FR_FR]
            },
            "favorites_count": len(self._favorites),
            "recently_used_count": len(self._recently_used),
        }


# =============================================================================
# Configuration Manager
# =============================================================================


class VoiceConfigurationManager:
    """Manager for voice configurations."""

    def __init__(
        self,
        stt_manager: STTProviderManager,
        tts_manager: TTSProviderManager,
    ):
        self._stt_manager = stt_manager
        self._tts_manager = tts_manager
        self._configurations: Dict[str, VoiceConfiguration] = {}
        self._by_organization: Dict[str, List[str]] = {}
        self._by_agent: Dict[str, str] = {}

    async def create_configuration(
        self,
        organization_id: str,
        agent_id: str,
        name: str,
        stt_provider: STTProvider = STTProvider.DEEPGRAM,
        stt_model_id: str = "nova-2",
        tts_provider: TTSProvider = TTSProvider.ELEVENLABS,
        tts_model_id: str = "eleven_turbo_v2_5",
        voice_id: str = "",
        custom_voice_id: Optional[str] = None,
        language: LanguageCode = LanguageCode.EN_US,
        **kwargs,
    ) -> VoiceConfiguration:
        """Create a new voice configuration."""
        config_id = f"voice_config_{uuid.uuid4().hex[:14]}"

        stt_config = STTConfiguration(
            id=f"stt_config_{uuid.uuid4().hex[:12]}",
            organization_id=organization_id,
            name=f"{name} STT",
            provider=stt_provider,
            model_id=stt_model_id,
            language=language,
        )

        tts_config = TTSConfiguration(
            id=f"tts_config_{uuid.uuid4().hex[:12]}",
            organization_id=organization_id,
            name=f"{name} TTS",
            provider=tts_provider,
            model_id=tts_model_id,
            voice_id=voice_id,
            custom_voice_id=custom_voice_id,
            language=language,
        )

        config = VoiceConfiguration(
            id=config_id,
            organization_id=organization_id,
            agent_id=agent_id,
            name=name,
            stt=stt_config,
            tts=tts_config,
            **kwargs,
        )

        self._configurations[config_id] = config

        # Index by organization
        if organization_id not in self._by_organization:
            self._by_organization[organization_id] = []
        self._by_organization[organization_id].append(config_id)

        # Index by agent
        self._by_agent[agent_id] = config_id

        return config

    async def get_configuration(
        self, config_id: str
    ) -> Optional[VoiceConfiguration]:
        """Get a configuration by ID."""
        return self._configurations.get(config_id)

    async def get_by_agent(self, agent_id: str) -> Optional[VoiceConfiguration]:
        """Get configuration for an agent."""
        config_id = self._by_agent.get(agent_id)
        if config_id:
            return self._configurations.get(config_id)
        return None

    async def get_by_organization(
        self, organization_id: str
    ) -> List[VoiceConfiguration]:
        """Get all configurations for an organization."""
        config_ids = self._by_organization.get(organization_id, [])
        return [self._configurations[cid] for cid in config_ids]

    async def update_configuration(
        self,
        config_id: str,
        **updates,
    ) -> Optional[VoiceConfiguration]:
        """Update a configuration."""
        config = self._configurations.get(config_id)
        if not config:
            return None

        # Handle nested updates
        stt_updates = updates.pop("stt", None)
        tts_updates = updates.pop("tts", None)

        # Update top-level fields
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Update STT config
        if stt_updates and config.stt:
            for key, value in stt_updates.items():
                if hasattr(config.stt, key):
                    setattr(config.stt, key, value)
            config.stt.updated_at = datetime.utcnow()

        # Update TTS config
        if tts_updates and config.tts:
            for key, value in tts_updates.items():
                if hasattr(config.tts, key):
                    setattr(config.tts, key, value)
            config.tts.updated_at = datetime.utcnow()

        config.updated_at = datetime.utcnow()

        return config

    async def delete_configuration(self, config_id: str) -> bool:
        """Delete a configuration."""
        config = self._configurations.get(config_id)
        if not config:
            return False

        # Remove from indexes
        if config.organization_id in self._by_organization:
            self._by_organization[config.organization_id].remove(config_id)
        if config.agent_id in self._by_agent:
            del self._by_agent[config.agent_id]

        del self._configurations[config_id]
        return True

    async def duplicate_configuration(
        self,
        config_id: str,
        new_name: str,
        new_agent_id: Optional[str] = None,
    ) -> Optional[VoiceConfiguration]:
        """Duplicate a configuration."""
        original = self._configurations.get(config_id)
        if not original:
            return None

        return await self.create_configuration(
            organization_id=original.organization_id,
            agent_id=new_agent_id or f"agent_{uuid.uuid4().hex[:12]}",
            name=new_name,
            stt_provider=original.stt.provider,
            stt_model_id=original.stt.model_id,
            tts_provider=original.tts.provider,
            tts_model_id=original.tts.model_id,
            voice_id=original.tts.voice_id,
            custom_voice_id=original.tts.custom_voice_id,
            language=original.stt.language,
            vad_enabled=original.vad_enabled,
            vad_threshold=original.vad_threshold,
            turn_detection_enabled=original.turn_detection_enabled,
            allow_interruption=original.allow_interruption,
            backchanneling_enabled=original.backchanneling_enabled,
            filler_enabled=original.filler_enabled,
            noise_suppression=original.noise_suppression,
            optimize_latency=original.optimize_latency,
            target_latency_ms=original.target_latency_ms,
        )

    def validate_configuration(
        self, config: VoiceConfiguration
    ) -> Tuple[bool, List[str]]:
        """
        Validate a configuration.

        Returns (is_valid, errors).
        """
        errors = []

        # Validate STT provider
        stt_provider = self._stt_manager.get_provider(config.stt.provider)
        if not stt_provider:
            errors.append(f"STT provider not found: {config.stt.provider}")
        elif not stt_provider.is_enabled:
            errors.append(f"STT provider not enabled: {config.stt.provider}")
        else:
            # Validate STT model
            model = self._stt_manager.get_model(
                config.stt.provider, config.stt.model_id
            )
            if not model:
                errors.append(
                    f"STT model not found: {config.stt.model_id} "
                    f"for provider {config.stt.provider}"
                )

        # Validate TTS provider
        tts_provider = self._tts_manager.get_provider(config.tts.provider)
        if not tts_provider:
            errors.append(f"TTS provider not found: {config.tts.provider}")
        elif not tts_provider.is_enabled:
            errors.append(f"TTS provider not enabled: {config.tts.provider}")
        else:
            # Validate TTS model
            model = self._tts_manager.get_model(
                config.tts.provider, config.tts.model_id
            )
            if not model:
                errors.append(
                    f"TTS model not found: {config.tts.model_id} "
                    f"for provider {config.tts.provider}"
                )

            # Validate voice (if not using custom)
            if not config.tts.custom_voice_id and config.tts.voice_id:
                voice = self._tts_manager.get_voice(
                    config.tts.provider, config.tts.voice_id
                )
                if not voice:
                    errors.append(
                        f"Voice not found: {config.tts.voice_id} "
                        f"for provider {config.tts.provider}"
                    )

        # Validate settings
        if config.tts.settings:
            if not 0.25 <= config.tts.settings.speed <= 4.0:
                errors.append("Speed must be between 0.25 and 4.0")
            if not 0.5 <= config.tts.settings.pitch <= 2.0:
                errors.append("Pitch must be between 0.5 and 2.0")
            if not 0.0 <= config.tts.settings.stability <= 1.0:
                errors.append("Stability must be between 0.0 and 1.0")

        return len(errors) == 0, errors


# =============================================================================
# Preset Manager
# =============================================================================


class PresetManager:
    """Manager for configuration presets."""

    def __init__(self):
        self._presets: Dict[str, VoiceConfigurationPreset] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load default presets."""
        for preset in get_default_presets():
            self._presets[preset.id] = preset

    def get_preset(self, preset_id: str) -> Optional[VoiceConfigurationPreset]:
        """Get a preset by ID."""
        return self._presets.get(preset_id)

    def get_all_presets(self) -> List[VoiceConfigurationPreset]:
        """Get all presets."""
        return list(self._presets.values())

    def get_by_category(self, category: str) -> List[VoiceConfigurationPreset]:
        """Get presets by category."""
        return [p for p in self._presets.values() if p.category == category]

    def get_recommended(self, use_case: str) -> List[VoiceConfigurationPreset]:
        """Get recommended presets for a use case."""
        use_case_lower = use_case.lower()
        result = []
        for preset in self._presets.values():
            if any(use_case_lower in r.lower() for r in preset.recommended_for):
                result.append(preset)
        return result

    async def create_preset(
        self,
        name: str,
        description: str,
        category: str = "custom",
        **kwargs,
    ) -> VoiceConfigurationPreset:
        """Create a custom preset."""
        preset = VoiceConfigurationPreset(
            id=f"preset_{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            category=category,
            is_system=False,
            **kwargs,
        )
        self._presets[preset.id] = preset
        return preset

    async def update_preset(
        self, preset_id: str, **updates
    ) -> Optional[VoiceConfigurationPreset]:
        """Update a preset (only non-system presets)."""
        preset = self._presets.get(preset_id)
        if not preset or preset.is_system:
            return None

        for key, value in updates.items():
            if hasattr(preset, key):
                setattr(preset, key, value)

        return preset

    async def delete_preset(self, preset_id: str) -> bool:
        """Delete a preset (only non-system presets)."""
        preset = self._presets.get(preset_id)
        if not preset or preset.is_system:
            return False

        del self._presets[preset_id]
        return True

    def apply_preset(
        self,
        preset_id: str,
        config: VoiceConfiguration,
    ) -> VoiceConfiguration:
        """Apply a preset to a configuration."""
        preset = self._presets.get(preset_id)
        if not preset:
            raise ValueError(f"Preset not found: {preset_id}")
        return preset.apply_to(config)


# =============================================================================
# Voice Preview Service
# =============================================================================


class VoicePreviewService:
    """Service for generating voice previews."""

    def __init__(self, tts_manager: TTSProviderManager):
        self._tts_manager = tts_manager
        self._cache: Dict[str, bytes] = {}
        self._cache_ttl: Dict[str, datetime] = {}

    async def generate_preview(
        self,
        provider: TTSProvider,
        voice_id: str,
        text: Optional[str] = None,
        settings: Optional[VoiceSettings] = None,
    ) -> bytes:
        """
        Generate a voice preview.

        Returns audio bytes.
        """
        # Default preview text
        if not text:
            text = "Hello, this is a sample of my voice. I can help you with various tasks."

        # Generate cache key
        cache_key = self._generate_cache_key(provider, voice_id, text, settings)

        # Check cache
        if cache_key in self._cache:
            if self._cache_ttl.get(cache_key, datetime.min) > datetime.utcnow():
                return self._cache[cache_key]

        # Generate preview (placeholder - actual implementation would call TTS API)
        # In real implementation, this would call the TTS provider's API
        audio_data = await self._synthesize(provider, voice_id, text, settings)

        # Cache result
        self._cache[cache_key] = audio_data
        self._cache_ttl[cache_key] = datetime.utcnow() + timedelta(hours=1)

        return audio_data

    async def _synthesize(
        self,
        provider: TTSProvider,
        voice_id: str,
        text: str,
        settings: Optional[VoiceSettings],
    ) -> bytes:
        """
        Synthesize speech using the TTS provider.

        This is a placeholder - actual implementation would call provider API.
        """
        # Placeholder implementation
        # In real implementation, this would:
        # 1. Get provider config
        # 2. Make API call to provider
        # 3. Return audio bytes

        # For now, return empty bytes
        return b""

    def _generate_cache_key(
        self,
        provider: TTSProvider,
        voice_id: str,
        text: str,
        settings: Optional[VoiceSettings],
    ) -> str:
        """Generate a cache key for the preview."""
        settings_str = str(settings.to_dict()) if settings else ""
        key_str = f"{provider.value}:{voice_id}:{text}:{settings_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear the preview cache."""
        self._cache.clear()
        self._cache_ttl.clear()


# =============================================================================
# Provider Health Checker
# =============================================================================


class ProviderHealthChecker:
    """Service for monitoring provider health."""

    def __init__(
        self,
        stt_manager: STTProviderManager,
        tts_manager: TTSProviderManager,
    ):
        self._stt_manager = stt_manager
        self._tts_manager = tts_manager
        self._check_interval = 60  # seconds
        self._last_check: Optional[datetime] = None

    async def check_all_providers(self) -> Dict[str, Dict[str, bool]]:
        """Check health of all providers."""
        results = {
            "stt": {},
            "tts": {},
        }

        # Check STT providers
        for provider in self._stt_manager.get_enabled_providers():
            is_healthy = await self._check_stt_provider(provider.provider)
            results["stt"][provider.provider.value] = is_healthy
            self._stt_manager.update_health(provider.provider, is_healthy)

        # Check TTS providers
        for provider in self._tts_manager.get_enabled_providers():
            is_healthy = await self._check_tts_provider(provider.provider)
            results["tts"][provider.provider.value] = is_healthy
            self._tts_manager.update_health(provider.provider, is_healthy)

        self._last_check = datetime.utcnow()

        return results

    async def _check_stt_provider(self, provider: STTProvider) -> bool:
        """Check STT provider health."""
        # Placeholder - actual implementation would make test API call
        return True

    async def _check_tts_provider(self, provider: TTSProvider) -> bool:
        """Check TTS provider health."""
        # Placeholder - actual implementation would make test API call
        return True

    async def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "stt_providers": {
                p.provider.value: {
                    "is_healthy": p.is_healthy,
                    "error_rate": p.error_rate,
                    "last_check": p.last_health_check.isoformat()
                    if p.last_health_check else None,
                }
                for p in self._stt_manager.get_all_providers()
            },
            "tts_providers": {
                p.provider.value: {
                    "is_healthy": p.is_healthy,
                    "error_rate": p.error_rate,
                    "last_check": p.last_health_check.isoformat()
                    if p.last_health_check else None,
                }
                for p in self._tts_manager.get_all_providers()
            },
            "last_full_check": self._last_check.isoformat()
            if self._last_check else None,
        }


# =============================================================================
# Main Voice Configuration Service
# =============================================================================


class VoiceConfigurationService:
    """
    Main service for voice configuration management.

    This is the primary entry point for all voice configuration operations,
    providing a unified API for:
    - Provider management (STT and TTS)
    - Voice library browsing and search
    - Configuration CRUD operations
    - Preset management
    - Voice previews
    - Health monitoring
    """

    def __init__(self):
        # Initialize managers
        self.stt_providers = STTProviderManager()
        self.tts_providers = TTSProviderManager()

        # Initialize services
        self.voice_library = VoiceLibrary(self.tts_providers)
        self.configurations = VoiceConfigurationManager(
            self.stt_providers, self.tts_providers
        )
        self.presets = PresetManager()
        self.previews = VoicePreviewService(self.tts_providers)
        self.health_checker = ProviderHealthChecker(
            self.stt_providers, self.tts_providers
        )

    # -------------------------------------------------------------------------
    # Provider Configuration
    # -------------------------------------------------------------------------

    def configure_stt_provider(
        self,
        provider: STTProvider,
        api_key: str,
        **kwargs,
    ) -> STTProviderConfig:
        """Configure an STT provider with API credentials."""
        return self.stt_providers.configure_provider(
            provider=provider,
            api_key=api_key,
            **kwargs,
        )

    def configure_tts_provider(
        self,
        provider: TTSProvider,
        api_key: str,
        **kwargs,
    ) -> TTSProviderConfig:
        """Configure a TTS provider with API credentials."""
        return self.tts_providers.configure_provider(
            provider=provider,
            api_key=api_key,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Voice Configuration CRUD
    # -------------------------------------------------------------------------

    async def create_configuration(
        self,
        organization_id: str,
        agent_id: str,
        name: str,
        preset_id: Optional[str] = None,
        **kwargs,
    ) -> VoiceConfiguration:
        """
        Create a new voice configuration.

        If preset_id is provided, the preset settings will be applied.
        """
        # Get defaults from preset if provided
        if preset_id:
            preset = self.presets.get_preset(preset_id)
            if preset:
                kwargs.setdefault("stt_provider", preset.stt_provider)
                kwargs.setdefault("stt_model_id", preset.stt_model_id)
                kwargs.setdefault("tts_provider", preset.tts_provider)
                kwargs.setdefault("tts_model_id", preset.tts_model_id)
                kwargs.setdefault("vad_enabled", preset.vad_enabled)
                kwargs.setdefault("turn_detection_enabled", preset.turn_detection_enabled)
                kwargs.setdefault("allow_interruption", preset.allow_interruption)
                kwargs.setdefault("backchanneling_enabled", preset.backchanneling_enabled)
                kwargs.setdefault("filler_enabled", preset.filler_enabled)
                kwargs.setdefault("noise_suppression", preset.noise_suppression)
                kwargs.setdefault("target_latency_ms", preset.target_latency_ms)

        return await self.configurations.create_configuration(
            organization_id=organization_id,
            agent_id=agent_id,
            name=name,
            **kwargs,
        )

    async def get_configuration(
        self, config_id: str
    ) -> Optional[VoiceConfiguration]:
        """Get a voice configuration by ID."""
        return await self.configurations.get_configuration(config_id)

    async def get_agent_configuration(
        self, agent_id: str
    ) -> Optional[VoiceConfiguration]:
        """Get the voice configuration for an agent."""
        return await self.configurations.get_by_agent(agent_id)

    async def update_configuration(
        self, config_id: str, **updates
    ) -> Optional[VoiceConfiguration]:
        """Update a voice configuration."""
        return await self.configurations.update_configuration(config_id, **updates)

    async def delete_configuration(self, config_id: str) -> bool:
        """Delete a voice configuration."""
        return await self.configurations.delete_configuration(config_id)

    # -------------------------------------------------------------------------
    # Voice Management
    # -------------------------------------------------------------------------

    def get_available_voices(
        self,
        provider: Optional[TTSProvider] = None,
        language: Optional[LanguageCode] = None,
        gender: Optional[VoiceGender] = None,
        style: Optional[VoiceStyle] = None,
    ) -> List[Voice]:
        """Get available voices with optional filtering."""
        filters = VoiceSearchFilters(
            providers=[provider] if provider else None,
            languages=[language] if language else None,
            genders=[gender] if gender else None,
            styles=[style] if style else None,
        )
        voices, _ = self.voice_library.search(filters)
        return voices

    def search_voices(
        self,
        query: str,
        provider: Optional[TTSProvider] = None,
        limit: int = 20,
    ) -> List[Voice]:
        """Search voices by text query."""
        filters = VoiceSearchFilters(
            query=query,
            providers=[provider] if provider else None,
        )
        voices, _ = self.voice_library.search(filters, limit=limit)
        return voices

    def add_custom_voice(
        self,
        provider: TTSProvider,
        voice_id: str,
        name: str,
        display_name: str,
        **kwargs,
    ) -> Voice:
        """Add a custom voice ID."""
        return self.tts_providers.add_custom_voice(
            provider=provider,
            voice_id=voice_id,
            name=name,
            display_name=display_name,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Preset Management
    # -------------------------------------------------------------------------

    def get_presets(
        self, category: Optional[str] = None
    ) -> List[VoiceConfigurationPreset]:
        """Get configuration presets."""
        if category:
            return self.presets.get_by_category(category)
        return self.presets.get_all_presets()

    async def apply_preset(
        self, config_id: str, preset_id: str
    ) -> Optional[VoiceConfiguration]:
        """Apply a preset to a configuration."""
        config = await self.configurations.get_configuration(config_id)
        if not config:
            return None

        return self.presets.apply_preset(preset_id, config)

    # -------------------------------------------------------------------------
    # Preview and Testing
    # -------------------------------------------------------------------------

    async def generate_voice_preview(
        self,
        provider: TTSProvider,
        voice_id: str,
        text: Optional[str] = None,
        settings: Optional[VoiceSettings] = None,
    ) -> bytes:
        """Generate a voice preview."""
        return await self.previews.generate_preview(
            provider=provider,
            voice_id=voice_id,
            text=text,
            settings=settings,
        )

    # -------------------------------------------------------------------------
    # Health and Status
    # -------------------------------------------------------------------------

    async def check_provider_health(self) -> Dict[str, Dict[str, bool]]:
        """Check health of all providers."""
        return await self.health_checker.check_all_providers()

    async def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        health_status = await self.health_checker.get_status()
        voice_stats = self.voice_library.get_statistics()

        return {
            "health": health_status,
            "voice_library": voice_stats,
            "stt_providers": len(self.stt_providers.get_enabled_providers()),
            "tts_providers": len(self.tts_providers.get_enabled_providers()),
            "presets": len(self.presets.get_all_presets()),
        }

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def validate_configuration(
        self, config: VoiceConfiguration
    ) -> Tuple[bool, List[str]]:
        """Validate a configuration."""
        return self.configurations.validate_configuration(config)

    def get_recommended_voices(
        self,
        use_case: str,
        language: LanguageCode = LanguageCode.EN_US,
    ) -> List[Voice]:
        """Get recommended voices for a use case."""
        return self.voice_library.get_recommended(use_case, language)

    def get_recommended_presets(self, use_case: str) -> List[VoiceConfigurationPreset]:
        """Get recommended presets for a use case."""
        return self.presets.get_recommended(use_case)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
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
]

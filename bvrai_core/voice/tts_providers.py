"""
BVRAI Voice Engine - TTS Provider Implementations

Multi-provider Text-to-Speech support with streaming capabilities,
voice cloning, and professional-grade audio synthesis.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from .base import (
    AudioChunk,
    AudioConfig,
    AudioFormat,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    QuotaExceededError,
    SynthesisResult,
    SynthesisStatus,
    TTSConfig,
    TTSError,
    TTSProvider,
    TTSProviderInterface,
    VoiceNotFoundError,
    generate_request_id,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ElevenLabs Provider
# =============================================================================

class ElevenLabsTTSProvider(TTSProviderInterface):
    """
    ElevenLabs Text-to-Speech provider.

    Industry-leading voice synthesis with voice cloning,
    multilingual support, and ultra-low latency streaming.
    """

    # Pricing per 1000 characters
    MODEL_PRICING = {
        "eleven_monolingual_v1": 0.30,
        "eleven_multilingual_v1": 0.30,
        "eleven_multilingual_v2": 0.30,
        "eleven_turbo_v2": 0.18,
        "eleven_turbo_v2_5": 0.18,
        "eleven_flash_v2": 0.08,
        "eleven_flash_v2_5": 0.08,
    }

    # Default voices
    DEFAULT_VOICES = {
        "rachel": "21m00Tcm4TlvDq8ikWAM",
        "drew": "29vD33N1CtxCmqQRPOHJ",
        "clyde": "2EiwWnXFnvU5JabPnv8n",
        "paul": "5Q0t7uMcjvnagumLfvZi",
        "domi": "AZnzlk1XvdvUeBnXmlld",
        "bella": "EXAVITQu4vr4xnSDxMaL",
        "antoni": "ErXwobaYiN019PkySvjV",
        "thomas": "GBv7mTt0atIp3Br8iCZE",
        "charlie": "IKne3meq5aSn9XLyUdCD",
        "emily": "LcfcDJNUP1GQjkzn1xUU",
        "elli": "MF3mGyEYCl7XYWbV9V6O",
        "callum": "N2lVS1w4EtoT3dr4eOWO",
        "patrick": "ODq5zmih8GrVes37Dizd",
        "harry": "SOYHLrjzK2X1ezoPC6cr",
        "liam": "TX3LPaxmHKxFdv7VOQHJ",
        "dorothy": "ThT5KcBeYPX3keUQqHPh",
        "josh": "TxGEqnHWrfWFTfGW9XjX",
        "arnold": "VR6AewLTigWG4xSOukaG",
        "charlotte": "XB0fDUnXU5powFXDhCwa",
        "matilda": "XrExE9yKIg1WjnnlVkGX",
        "matthew": "Yko7PKs4b5dXpnhwxyPK",
        "james": "ZQe5CZNOzWyzPSCn5a3c",
        "joseph": "Zlb1dXrM653N07WRdFW3",
        "jeremy": "bVMeCyTHy58xNoL34h3p",
        "michael": "flq6f7yk4E4fJM5XTYuZ",
        "ethan": "g5CIjZEefAph4nQFvHAz",
        "gigi": "jBpfuIE2acCO8z3wKNLl",
        "freya": "jsCqWAovK2LkecY7zXl4",
        "grace": "oWAxZDx7w5VEj9dCyTzz",
        "daniel": "onwK4e9ZLuTAKqWW03F9",
        "serena": "pMsXgVXv3BLzUgSXRplE",
        "adam": "pNInz6obpgDQGcFmaJgB",
        "nicole": "piTKgcLEGmPE4e6mEKli",
        "jessie": "t0jbNlBVZ17f02VDIeMI",
        "ryan": "wViXBPUzp2ZZixB1xQuM",
        "sam": "yoZ06aMxZJJ28mfd3POQ",
        "glinda": "z9fAnlkpzviPz146aGWa",
        "giovanni": "zcAOhNBS3c14rBihAFp1",
        "mimi": "zrHiDhphv9ZnVXBqCLjz",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.elevenlabs.io/v1",
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")

        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._voices_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_time: Optional[datetime] = None

    @property
    def provider_name(self) -> TTSProvider:
        return TTSProvider.ELEVENLABS

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def synthesize(
        self,
        text: str,
        config: TTSConfig
    ) -> SynthesisResult:
        """Synthesize text to speech using ElevenLabs."""
        request_id = generate_request_id()
        start_time = time.monotonic()

        if not text.strip():
            return SynthesisResult(
                id=request_id,
                audio_data=b"",
                status=SynthesisStatus.COMPLETED,
                text_length=0,
                provider=TTSProvider.ELEVENLABS,
            )

        try:
            client = await self._get_client()

            # Prepare request body
            body = {
                "text": text,
                "model_id": config.model or "eleven_turbo_v2",
                "voice_settings": {
                    "stability": config.stability,
                    "similarity_boost": config.similarity_boost,
                    "style": config.style,
                    "use_speaker_boost": config.use_speaker_boost,
                },
            }

            # Determine output format
            output_format = self._get_output_format(config)

            response = await client.post(
                f"/text-to-speech/{config.voice_id}",
                json=body,
                params={"output_format": output_format},
            )

            latency_ms = (time.monotonic() - start_time) * 1000

            if response.status_code == 401:
                raise TTSError(
                    "Invalid ElevenLabs API key",
                    provider="elevenlabs"
                )

            if response.status_code == 404:
                raise VoiceNotFoundError(
                    f"Voice '{config.voice_id}' not found",
                    voice_id=config.voice_id,
                    provider="elevenlabs"
                )

            if response.status_code == 429:
                raise ProviderRateLimitError(
                    "ElevenLabs rate limit exceeded",
                    provider="elevenlabs",
                    retry_after=int(response.headers.get("Retry-After", 60))
                )

            if response.status_code == 402:
                raise QuotaExceededError(
                    "ElevenLabs quota exceeded. Please upgrade your plan.",
                    provider="elevenlabs"
                )

            if response.status_code != 200:
                raise TTSError(
                    f"ElevenLabs API error: {response.status_code} - {response.text}",
                    provider="elevenlabs"
                )

            audio_data = response.content

            # Calculate cost
            characters = len(text)
            price_per_1k = self.MODEL_PRICING.get(config.model, 0.18)
            estimated_cost = (characters / 1000) * price_per_1k

            # Estimate duration (rough: ~150 words/min, ~5 chars/word)
            estimated_duration = (characters / 5) / 150 * 60

            return SynthesisResult(
                id=request_id,
                audio_data=audio_data,
                status=SynthesisStatus.COMPLETED,
                text_length=characters,
                audio_duration=estimated_duration,
                audio_format=config.audio_config.format,
                sample_rate=config.audio_config.sample_rate,
                provider=TTSProvider.ELEVENLABS,
                voice_id=config.voice_id,
                model=config.model,
                latency_ms=latency_ms,
                processing_time_ms=latency_ms,
                characters_billed=characters,
                estimated_cost=estimated_cost,
            )

        except (TTSError, VoiceNotFoundError, ProviderRateLimitError, QuotaExceededError):
            raise
        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                f"ElevenLabs request timed out after {self.timeout}s",
                provider="elevenlabs"
            )
        except httpx.ConnectError as e:
            raise ProviderConnectionError(
                "Failed to connect to ElevenLabs API",
                provider="elevenlabs",
                details={"error": str(e)}
            )
        except Exception as e:
            logger.exception(f"ElevenLabs synthesis error: {e}")
            raise TTSError(
                f"Synthesis failed: {str(e)}",
                provider="elevenlabs"
            )

    async def synthesize_stream(
        self,
        text: str,
        config: TTSConfig
    ) -> AsyncGenerator[AudioChunk, None]:
        """Stream audio synthesis from ElevenLabs."""
        request_id = generate_request_id()
        start_time = time.monotonic()

        if not text.strip():
            return

        try:
            client = await self._get_client()

            body = {
                "text": text,
                "model_id": config.model or "eleven_turbo_v2",
                "voice_settings": {
                    "stability": config.stability,
                    "similarity_boost": config.similarity_boost,
                    "style": config.style,
                    "use_speaker_boost": config.use_speaker_boost,
                },
            }

            output_format = self._get_output_format(config)

            # Use streaming endpoint
            async with client.stream(
                "POST",
                f"/text-to-speech/{config.voice_id}/stream",
                json=body,
                params={
                    "output_format": output_format,
                    "optimize_streaming_latency": config.optimize_streaming_latency,
                },
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise TTSError(
                        f"ElevenLabs streaming error: {response.status_code}",
                        provider="elevenlabs"
                    )

                sequence = 0
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    sequence += 1
                    yield AudioChunk(
                        data=chunk,
                        sequence=sequence,
                        timestamp=time.monotonic() - start_time,
                        is_final=False,
                        duration_ms=len(chunk) / 32,  # Rough estimate
                    )

                # Send final chunk marker
                yield AudioChunk(
                    data=b"",
                    sequence=sequence + 1,
                    timestamp=time.monotonic() - start_time,
                    is_final=True,
                )

        except TTSError:
            raise
        except Exception as e:
            logger.exception(f"ElevenLabs streaming error: {e}")
            raise TTSError(
                f"Streaming synthesis failed: {str(e)}",
                provider="elevenlabs"
            )

    async def get_voices(self) -> List[Dict[str, Any]]:
        """Get available voices from ElevenLabs."""
        # Check cache (5 minute TTL)
        if self._voices_cache and self._cache_time:
            age = (datetime.utcnow() - self._cache_time).total_seconds()
            if age < 300:
                return self._voices_cache

        try:
            client = await self._get_client()
            response = await client.get("/voices")

            if response.status_code != 200:
                raise TTSError(
                    f"Failed to get voices: {response.status_code}",
                    provider="elevenlabs"
                )

            data = response.json()
            voices = data.get("voices", [])

            # Format voice data
            formatted_voices = []
            for voice in voices:
                formatted_voices.append({
                    "id": voice.get("voice_id"),
                    "name": voice.get("name"),
                    "category": voice.get("category"),
                    "description": voice.get("description"),
                    "preview_url": voice.get("preview_url"),
                    "labels": voice.get("labels", {}),
                    "settings": voice.get("settings", {}),
                })

            self._voices_cache = formatted_voices
            self._cache_time = datetime.utcnow()

            return formatted_voices

        except TTSError:
            raise
        except Exception as e:
            logger.exception(f"Error getting ElevenLabs voices: {e}")
            raise TTSError(
                f"Failed to get voices: {str(e)}",
                provider="elevenlabs"
            )

    def _get_output_format(self, config: TTSConfig) -> str:
        """Get ElevenLabs output format string."""
        format_map = {
            AudioFormat.MP3: f"mp3_{config.audio_config.sample_rate}_128",
            AudioFormat.PCM_16: f"pcm_{config.audio_config.sample_rate}",
            AudioFormat.OGG_OPUS: f"opus_{config.audio_config.sample_rate}",
            AudioFormat.MULAW: f"ulaw_{config.audio_config.sample_rate}",
        }
        return format_map.get(config.audio_config.format, "mp3_24000_128")

    async def health_check(self) -> bool:
        """Check ElevenLabs API health."""
        try:
            client = await self._get_client()
            response = await client.get("/user", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# =============================================================================
# OpenAI TTS Provider
# =============================================================================

class OpenAITTSProvider(TTSProviderInterface):
    """
    OpenAI Text-to-Speech provider.

    High-quality neural TTS with simple API.
    Supports multiple voices and HD quality.
    """

    # Available voices
    VOICES = {
        "alloy": "Versatile, balanced voice",
        "echo": "Warm, conversational voice",
        "fable": "British accent, expressive",
        "onyx": "Deep, authoritative voice",
        "nova": "Friendly, conversational female voice",
        "shimmer": "Warm, engaging voice",
    }

    # Pricing per 1M characters
    MODEL_PRICING = {
        "tts-1": 15.00,      # $15 per 1M chars
        "tts-1-hd": 30.00,   # $30 per 1M chars
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_name(self) -> TTSProvider:
        return TTSProvider.OPENAI

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def synthesize(
        self,
        text: str,
        config: TTSConfig
    ) -> SynthesisResult:
        """Synthesize text to speech using OpenAI TTS."""
        request_id = generate_request_id()
        start_time = time.monotonic()

        if not text.strip():
            return SynthesisResult(
                id=request_id,
                audio_data=b"",
                status=SynthesisStatus.COMPLETED,
                text_length=0,
                provider=TTSProvider.OPENAI,
            )

        # OpenAI TTS has 4096 character limit
        if len(text) > 4096:
            logger.warning(f"Text truncated from {len(text)} to 4096 characters")
            text = text[:4096]

        try:
            client = await self._get_client()

            # Map voice_id to OpenAI voice name
            voice = config.voice_id if config.voice_id in self.VOICES else "alloy"

            body = {
                "model": config.model or "tts-1",
                "input": text,
                "voice": voice,
                "response_format": self._get_response_format(config),
                "speed": config.speed,
            }

            response = await client.post("/audio/speech", json=body)

            latency_ms = (time.monotonic() - start_time) * 1000

            if response.status_code == 429:
                raise ProviderRateLimitError(
                    "OpenAI rate limit exceeded",
                    provider="openai"
                )

            if response.status_code != 200:
                raise TTSError(
                    f"OpenAI TTS error: {response.status_code} - {response.text}",
                    provider="openai"
                )

            audio_data = response.content

            # Calculate cost
            characters = len(text)
            price_per_1m = self.MODEL_PRICING.get(config.model, 15.00)
            estimated_cost = (characters / 1_000_000) * price_per_1m

            return SynthesisResult(
                id=request_id,
                audio_data=audio_data,
                status=SynthesisStatus.COMPLETED,
                text_length=characters,
                audio_format=config.audio_config.format,
                provider=TTSProvider.OPENAI,
                voice_id=voice,
                model=config.model or "tts-1",
                latency_ms=latency_ms,
                characters_billed=characters,
                estimated_cost=estimated_cost,
            )

        except (TTSError, ProviderRateLimitError):
            raise
        except Exception as e:
            logger.exception(f"OpenAI TTS error: {e}")
            raise TTSError(
                f"Synthesis failed: {str(e)}",
                provider="openai"
            )

    async def synthesize_stream(
        self,
        text: str,
        config: TTSConfig
    ) -> AsyncGenerator[AudioChunk, None]:
        """Stream audio from OpenAI TTS."""
        request_id = generate_request_id()
        start_time = time.monotonic()

        if not text.strip():
            return

        if len(text) > 4096:
            text = text[:4096]

        try:
            client = await self._get_client()

            voice = config.voice_id if config.voice_id in self.VOICES else "alloy"

            body = {
                "model": config.model or "tts-1",
                "input": text,
                "voice": voice,
                "response_format": "pcm",  # PCM for streaming
                "speed": config.speed,
            }

            async with client.stream("POST", "/audio/speech", json=body) as response:
                if response.status_code != 200:
                    raise TTSError(
                        f"OpenAI TTS streaming error: {response.status_code}",
                        provider="openai"
                    )

                sequence = 0
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    sequence += 1
                    yield AudioChunk(
                        data=chunk,
                        sequence=sequence,
                        timestamp=time.monotonic() - start_time,
                        is_final=False,
                    )

                yield AudioChunk(
                    data=b"",
                    sequence=sequence + 1,
                    timestamp=time.monotonic() - start_time,
                    is_final=True,
                )

        except TTSError:
            raise
        except Exception as e:
            logger.exception(f"OpenAI TTS streaming error: {e}")
            raise TTSError(
                f"Streaming failed: {str(e)}",
                provider="openai"
            )

    async def get_voices(self) -> List[Dict[str, Any]]:
        """Get available OpenAI TTS voices."""
        return [
            {
                "id": voice_id,
                "name": voice_id.capitalize(),
                "description": description,
                "provider": "openai",
            }
            for voice_id, description in self.VOICES.items()
        ]

    def _get_response_format(self, config: TTSConfig) -> str:
        """Get OpenAI response format string."""
        format_map = {
            AudioFormat.MP3: "mp3",
            AudioFormat.OGG_OPUS: "opus",
            AudioFormat.AAC: "aac",
            AudioFormat.FLAC: "flac",
            AudioFormat.WAV: "wav",
            AudioFormat.PCM_16: "pcm",
        }
        return format_map.get(config.audio_config.format, "mp3")

    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            client = await self._get_client()
            response = await client.get("/models", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# =============================================================================
# PlayHT Provider
# =============================================================================

class PlayHTTTSProvider(TTSProviderInterface):
    """
    PlayHT Text-to-Speech provider.

    Ultra-realistic AI voices with voice cloning.
    Low latency streaming for conversational AI.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        base_url: str = "https://api.play.ht/api/v2",
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.getenv("PLAYHT_API_KEY")
        self.user_id = user_id or os.getenv("PLAYHT_USER_ID")

        if not self.api_key or not self.user_id:
            raise ValueError("PlayHT API key and user ID are required")

        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_name(self) -> TTSProvider:
        return TTSProvider.PLAYHT

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "X-User-Id": self.user_id,
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def synthesize(
        self,
        text: str,
        config: TTSConfig
    ) -> SynthesisResult:
        """Synthesize text using PlayHT."""
        request_id = generate_request_id()
        start_time = time.monotonic()

        if not text.strip():
            return SynthesisResult(
                id=request_id,
                audio_data=b"",
                status=SynthesisStatus.COMPLETED,
                provider=TTSProvider.PLAYHT,
            )

        try:
            client = await self._get_client()

            body = {
                "text": text,
                "voice": config.voice_id,
                "output_format": "mp3",
                "speed": config.speed,
                "sample_rate": config.audio_config.sample_rate,
            }

            response = await client.post("/tts", json=body)

            latency_ms = (time.monotonic() - start_time) * 1000

            if response.status_code != 200:
                raise TTSError(
                    f"PlayHT error: {response.status_code} - {response.text}",
                    provider="playht"
                )

            # PlayHT returns URL to audio
            data = response.json()
            audio_url = data.get("audioUrl")

            if not audio_url:
                raise TTSError("No audio URL in PlayHT response", provider="playht")

            # Download audio
            audio_response = await client.get(audio_url)
            audio_data = audio_response.content

            return SynthesisResult(
                id=request_id,
                audio_data=audio_data,
                status=SynthesisStatus.COMPLETED,
                text_length=len(text),
                provider=TTSProvider.PLAYHT,
                voice_id=config.voice_id,
                latency_ms=latency_ms,
            )

        except TTSError:
            raise
        except Exception as e:
            logger.exception(f"PlayHT error: {e}")
            raise TTSError(f"Synthesis failed: {str(e)}", provider="playht")

    async def synthesize_stream(
        self,
        text: str,
        config: TTSConfig
    ) -> AsyncGenerator[AudioChunk, None]:
        """Stream synthesis from PlayHT."""
        request_id = generate_request_id()
        start_time = time.monotonic()

        if not text.strip():
            return

        try:
            client = await self._get_client()

            body = {
                "text": text,
                "voice": config.voice_id,
                "output_format": "raw",
                "speed": config.speed,
            }

            async with client.stream("POST", "/tts/stream", json=body) as response:
                if response.status_code != 200:
                    raise TTSError(
                        f"PlayHT streaming error: {response.status_code}",
                        provider="playht"
                    )

                sequence = 0
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    sequence += 1
                    yield AudioChunk(
                        data=chunk,
                        sequence=sequence,
                        timestamp=time.monotonic() - start_time,
                        is_final=False,
                    )

                yield AudioChunk(
                    data=b"",
                    sequence=sequence + 1,
                    timestamp=time.monotonic() - start_time,
                    is_final=True,
                )

        except TTSError:
            raise
        except Exception as e:
            raise TTSError(f"Streaming failed: {str(e)}", provider="playht")

    async def get_voices(self) -> List[Dict[str, Any]]:
        """Get PlayHT voices."""
        try:
            client = await self._get_client()
            response = await client.get("/voices")

            if response.status_code != 200:
                raise TTSError("Failed to get voices", provider="playht")

            return response.json()
        except Exception as e:
            raise TTSError(f"Failed to get voices: {str(e)}", provider="playht")

    async def health_check(self) -> bool:
        """Check PlayHT API health."""
        try:
            client = await self._get_client()
            response = await client.get("/voices", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# =============================================================================
# Mock Provider (for testing)
# =============================================================================

class MockTTSProvider(TTSProviderInterface):
    """Mock TTS provider for testing."""

    def __init__(self, latency_ms: float = 100.0):
        self.latency_ms = latency_ms
        self._call_count = 0

    @property
    def provider_name(self) -> TTSProvider:
        return TTSProvider.MOCK

    async def synthesize(
        self,
        text: str,
        config: TTSConfig
    ) -> SynthesisResult:
        """Return mock audio."""
        await asyncio.sleep(self.latency_ms / 1000)
        self._call_count += 1

        # Generate fake audio data (silence)
        duration = len(text) / 15  # ~15 chars per second
        sample_rate = config.audio_config.sample_rate
        num_samples = int(duration * sample_rate)
        audio_data = bytes(num_samples * 2)  # 16-bit silence

        return SynthesisResult(
            id=generate_request_id(),
            audio_data=audio_data,
            status=SynthesisStatus.COMPLETED,
            text_length=len(text),
            audio_duration=duration,
            provider=TTSProvider.MOCK,
            voice_id=config.voice_id,
            latency_ms=self.latency_ms,
        )

    async def synthesize_stream(
        self,
        text: str,
        config: TTSConfig
    ) -> AsyncGenerator[AudioChunk, None]:
        """Stream mock audio chunks."""
        chunk_size = 4096
        total_duration = len(text) / 15
        total_bytes = int(total_duration * config.audio_config.sample_rate * 2)

        sequence = 0
        start_time = time.monotonic()

        for offset in range(0, total_bytes, chunk_size):
            await asyncio.sleep(self.latency_ms / 1000 / 10)
            sequence += 1

            remaining = min(chunk_size, total_bytes - offset)
            yield AudioChunk(
                data=bytes(remaining),
                sequence=sequence,
                timestamp=time.monotonic() - start_time,
                is_final=False,
            )

        yield AudioChunk(
            data=b"",
            sequence=sequence + 1,
            timestamp=time.monotonic() - start_time,
            is_final=True,
        )

    async def get_voices(self) -> List[Dict[str, Any]]:
        """Return mock voices."""
        return [
            {"id": "mock-voice-1", "name": "Mock Voice 1"},
            {"id": "mock-voice-2", "name": "Mock Voice 2"},
        ]

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


# =============================================================================
# Provider Factory
# =============================================================================

def create_tts_provider(
    provider: TTSProvider,
    **kwargs
) -> TTSProviderInterface:
    """
    Factory function to create TTS providers.

    Args:
        provider: The provider type to create
        **kwargs: Provider-specific configuration

    Returns:
        Configured TTS provider instance
    """
    providers = {
        TTSProvider.ELEVENLABS: ElevenLabsTTSProvider,
        TTSProvider.OPENAI: OpenAITTSProvider,
        TTSProvider.PLAYHT: PlayHTTTSProvider,
        TTSProvider.MOCK: MockTTSProvider,
    }

    if provider not in providers:
        raise ValueError(f"Unsupported TTS provider: {provider}")

    return providers[provider](**kwargs)

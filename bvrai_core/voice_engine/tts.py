"""
Text-to-Speech (TTS) module.

Provides a unified interface for multiple TTS providers:
- ElevenLabs: Ultra-realistic voices, voice cloning
- PlayHT: High-quality, fast synthesis
- OpenAI: Natural TTS voices
- Azure: Enterprise-grade, many voices
- Google Cloud: Enterprise-grade, many languages

Features:
- Streaming audio synthesis
- Voice selection and configuration
- SSML support
- Prosody control (rate, pitch, volume)
- Caching for repeated phrases
- Voice cloning support (provider-dependent)
"""

import asyncio
import base64
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
import aiohttp
import websockets

from .audio import AudioFormat, AudioChunk, AudioCodec

logger = logging.getLogger(__name__)


class TTSProviderType(str, Enum):
    """Supported TTS providers."""
    ELEVENLABS = "elevenlabs"
    PLAYHT = "playht"
    OPENAI = "openai"
    AZURE = "azure"
    GOOGLE = "google"
    AMAZON = "amazon"


class VoiceGender(str, Enum):
    """Voice gender options."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceStyle(str, Enum):
    """Voice speaking styles."""
    NEUTRAL = "neutral"
    CHEERFUL = "cheerful"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    EMPATHETIC = "empathetic"
    CUSTOMER_SERVICE = "customer_service"
    NEWSCAST = "newscast"
    CONVERSATIONAL = "conversational"


@dataclass
class TTSVoice:
    """
    Voice configuration for TTS.

    Attributes:
        voice_id: Provider-specific voice identifier
        name: Human-readable voice name
        language: Language code (e.g., 'en-US')
        gender: Voice gender
        provider: TTS provider this voice belongs to
        sample_url: URL to voice sample audio
        styles: Available speaking styles
        is_cloned: Whether this is a cloned voice
    """
    voice_id: str
    name: str = ""
    language: str = "en-US"
    gender: VoiceGender = VoiceGender.NEUTRAL
    provider: str = ""
    sample_url: str = ""
    styles: List[VoiceStyle] = field(default_factory=list)
    is_cloned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "name": self.name,
            "language": self.language,
            "gender": self.gender.value,
            "provider": self.provider,
            "sample_url": self.sample_url,
            "styles": [s.value for s in self.styles],
            "is_cloned": self.is_cloned,
            "metadata": self.metadata,
        }


@dataclass
class TTSConfig:
    """
    Configuration for TTS synthesis.

    Attributes:
        voice: Voice to use
        model: Model to use (provider-specific)
        output_format: Desired output audio format

        speaking_rate: Speech rate (0.5 - 2.0, 1.0 = normal)
        pitch: Voice pitch adjustment (-20 to +20 semitones)
        volume: Volume adjustment (dB, -6 to +6)

        style: Speaking style
        style_degree: How strongly to apply style (0.0 - 2.0)

        stability: Voice stability (ElevenLabs specific)
        similarity_boost: Voice similarity (ElevenLabs specific)

        use_ssml: Whether input may contain SSML
        language: Output language (for multilingual voices)
    """
    voice: Optional[TTSVoice] = None
    voice_id: str = ""  # Alternative to voice object
    model: str = ""

    # Output format
    output_format: str = "mp3_44100_128"  # Provider-specific format string
    sample_rate: int = 24000
    encoding: str = "mp3"

    # Prosody
    speaking_rate: float = 1.0
    pitch: float = 0.0
    volume: float = 0.0

    # Style
    style: VoiceStyle = VoiceStyle.NEUTRAL
    style_degree: float = 1.0

    # ElevenLabs specific
    stability: float = 0.5
    similarity_boost: float = 0.75
    use_speaker_boost: bool = True

    # General
    use_ssml: bool = False
    language: str = ""

    # Streaming
    enable_streaming: bool = True
    chunk_size_ms: int = 100

    # Caching
    enable_caching: bool = True

    # Extra provider-specific options
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voice_id": self.voice.voice_id if self.voice else self.voice_id,
            "model": self.model,
            "output_format": self.output_format,
            "sample_rate": self.sample_rate,
            "speaking_rate": self.speaking_rate,
            "pitch": self.pitch,
            "volume": self.volume,
            "style": self.style.value,
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
        }


@dataclass
class TTSResult:
    """
    Result of TTS synthesis.

    Attributes:
        audio_data: Synthesized audio bytes
        format: Audio format
        duration_ms: Audio duration in milliseconds
        text: Original input text
        latency_ms: Time to first byte
    """
    audio_data: bytes = b""
    format: AudioFormat = field(default_factory=AudioFormat.wideband)
    duration_ms: float = 0.0
    text: str = ""
    latency_ms: float = 0.0
    provider: str = ""
    characters: int = 0

    def to_audio_chunk(self) -> AudioChunk:
        """Convert to AudioChunk."""
        return AudioChunk(
            data=self.audio_data,
            format=self.format,
        )


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""

    def __init__(self, api_key: str, config: TTSConfig):
        self.api_key = api_key
        self.config = config
        self._cache: Dict[str, bytes] = {}
        self._cache_max_size = 1000

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of this TTS provider."""
        pass

    @abstractmethod
    async def synthesize(self, text: str) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            TTSResult with audio data
        """
        pass

    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncIterator[AudioChunk]:
        """
        Synthesize speech and stream audio chunks.

        Args:
            text: Text to synthesize

        Yields:
            AudioChunk objects as synthesis progresses
        """
        pass

    @abstractmethod
    async def list_voices(self, language: str = "") -> List[TTSVoice]:
        """
        List available voices.

        Args:
            language: Filter by language code

        Returns:
            List of available voices
        """
        pass

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        key = f"{text}:{config_str}"
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cached(self, text: str) -> Optional[bytes]:
        """Get cached audio for text."""
        if not self.config.enable_caching:
            return None
        key = self._get_cache_key(text)
        return self._cache.get(key)

    def _set_cached(self, text: str, audio_data: bytes) -> None:
        """Cache audio for text."""
        if not self.config.enable_caching:
            return

        # Evict old entries if cache is full
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._cache.keys())[:100]
            for key in keys_to_remove:
                del self._cache[key]

        key = self._get_cache_key(text)
        self._cache[key] = audio_data

    def clear_cache(self) -> None:
        """Clear the audio cache."""
        self._cache.clear()


class ElevenLabsTTS(TTSProvider):
    """
    ElevenLabs TTS provider.

    Features:
    - Ultra-realistic AI voices
    - Voice cloning
    - Low-latency streaming
    - Multiple models (Turbo, Multilingual)
    """

    API_URL = "https://api.elevenlabs.io/v1"
    STREAMING_URL = "wss://api.elevenlabs.io/v1/text-to-speech"

    DEFAULT_VOICES = {
        "rachel": {"id": "21m00Tcm4TlvDq8ikWAM", "gender": "female"},
        "adam": {"id": "pNInz6obpgDQGcFmaJgB", "gender": "male"},
        "josh": {"id": "TxGEqnHWrfWFTfGW9XjX", "gender": "male"},
        "bella": {"id": "EXAVITQu4vr4xnSDxMaL", "gender": "female"},
        "elli": {"id": "MF3mGyEYCl7XYWbV9V6O", "gender": "female"},
        "sam": {"id": "yoZ06aMxZJJ28mfd3POQ", "gender": "male"},
    }

    def __init__(self, api_key: str, config: TTSConfig):
        super().__init__(api_key, config)

    @property
    def provider_name(self) -> str:
        return "elevenlabs"

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech using ElevenLabs API."""
        # Check cache
        cached = self._get_cached(text)
        if cached:
            return TTSResult(
                audio_data=cached,
                text=text,
                provider=self.provider_name,
                characters=len(text),
            )

        voice_id = self.config.voice.voice_id if self.config.voice else self.config.voice_id
        if not voice_id:
            voice_id = self.DEFAULT_VOICES["rachel"]["id"]

        model_id = self.config.model or "eleven_turbo_v2"

        url = f"{self.API_URL}/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost,
                "use_speaker_boost": self.config.use_speaker_boost,
            },
        }

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"ElevenLabs error: {error}")

                audio_data = await response.read()
                latency = (time.time() - start_time) * 1000

                # Cache the result
                self._set_cached(text, audio_data)

                return TTSResult(
                    audio_data=audio_data,
                    format=AudioFormat(
                        sample_rate=self.config.sample_rate,
                        codec=AudioCodec.MP3,
                    ),
                    text=text,
                    latency_ms=latency,
                    provider=self.provider_name,
                    characters=len(text),
                )

    async def synthesize_stream(self, text: str) -> AsyncIterator[AudioChunk]:
        """Stream synthesis using ElevenLabs WebSocket API."""
        voice_id = self.config.voice.voice_id if self.config.voice else self.config.voice_id
        if not voice_id:
            voice_id = self.DEFAULT_VOICES["rachel"]["id"]

        model_id = self.config.model or "eleven_turbo_v2"

        url = f"{self.STREAMING_URL}/{voice_id}/stream-input?model_id={model_id}"

        headers = {
            "xi-api-key": self.api_key,
        }

        try:
            async with websockets.connect(url, extra_headers=headers) as ws:
                # Send initial configuration
                await ws.send(json.dumps({
                    "text": " ",  # Initial space to prime the stream
                    "voice_settings": {
                        "stability": self.config.stability,
                        "similarity_boost": self.config.similarity_boost,
                    },
                    "generation_config": {
                        "chunk_length_schedule": [120, 160, 250, 290],
                    },
                }))

                # Send text
                await ws.send(json.dumps({
                    "text": text,
                }))

                # Send end of input
                await ws.send(json.dumps({
                    "text": "",
                }))

                # Receive audio chunks
                sequence = 0
                while True:
                    try:
                        message = await ws.recv()
                        data = json.loads(message)

                        if data.get("audio"):
                            audio_bytes = base64.b64decode(data["audio"])
                            yield AudioChunk(
                                data=audio_bytes,
                                format=AudioFormat(
                                    sample_rate=self.config.sample_rate,
                                    codec=AudioCodec.MP3,
                                ),
                                sequence=sequence,
                            )
                            sequence += 1

                        if data.get("isFinal"):
                            break

                    except websockets.exceptions.ConnectionClosed:
                        break

        except Exception as e:
            logger.error(f"ElevenLabs streaming error: {e}")
            # Fallback to non-streaming
            result = await self.synthesize(text)
            yield result.to_audio_chunk()

    async def list_voices(self, language: str = "") -> List[TTSVoice]:
        """List available ElevenLabs voices."""
        url = f"{self.API_URL}/voices"

        headers = {
            "xi-api-key": self.api_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                voices = []

                for voice in data.get("voices", []):
                    labels = voice.get("labels", {})
                    voice_lang = labels.get("language", "en")

                    if language and not voice_lang.startswith(language[:2]):
                        continue

                    gender = VoiceGender.NEUTRAL
                    if labels.get("gender") == "male":
                        gender = VoiceGender.MALE
                    elif labels.get("gender") == "female":
                        gender = VoiceGender.FEMALE

                    voices.append(TTSVoice(
                        voice_id=voice["voice_id"],
                        name=voice["name"],
                        language=voice_lang,
                        gender=gender,
                        provider=self.provider_name,
                        sample_url=voice.get("preview_url", ""),
                        is_cloned=voice.get("category") == "cloned",
                        metadata=labels,
                    ))

                return voices

    async def clone_voice(
        self,
        name: str,
        audio_files: List[bytes],
        description: str = "",
    ) -> TTSVoice:
        """Clone a voice from audio samples."""
        url = f"{self.API_URL}/voices/add"

        form = aiohttp.FormData()
        form.add_field("name", name)
        form.add_field("description", description)

        for i, audio in enumerate(audio_files):
            form.add_field(
                f"files",
                audio,
                filename=f"sample_{i}.mp3",
                content_type="audio/mpeg",
            )

        headers = {
            "xi-api-key": self.api_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=form) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Voice cloning failed: {error}")

                data = await response.json()

                return TTSVoice(
                    voice_id=data["voice_id"],
                    name=name,
                    provider=self.provider_name,
                    is_cloned=True,
                )


class PlayHTTTS(TTSProvider):
    """
    PlayHT TTS provider.

    Features:
    - High-quality voices
    - Fast synthesis
    - Voice cloning
    - Streaming support
    """

    API_URL = "https://api.play.ht/api/v2"

    def __init__(self, api_key: str, config: TTSConfig, user_id: str = ""):
        super().__init__(api_key, config)
        self.user_id = user_id

    @property
    def provider_name(self) -> str:
        return "playht"

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech using PlayHT API."""
        cached = self._get_cached(text)
        if cached:
            return TTSResult(
                audio_data=cached,
                text=text,
                provider=self.provider_name,
            )

        url = f"{self.API_URL}/tts"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-User-ID": self.user_id,
            "Content-Type": "application/json",
        }

        voice_id = self.config.voice.voice_id if self.config.voice else self.config.voice_id

        payload = {
            "text": text,
            "voice": voice_id,
            "output_format": self.config.output_format or "mp3",
            "speed": self.config.speaking_rate,
        }

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"PlayHT error: {error}")

                data = await response.json()
                audio_url = data.get("url")

                # Download audio
                async with session.get(audio_url) as audio_response:
                    audio_data = await audio_response.read()

                latency = (time.time() - start_time) * 1000

                self._set_cached(text, audio_data)

                return TTSResult(
                    audio_data=audio_data,
                    format=AudioFormat(sample_rate=self.config.sample_rate, codec=AudioCodec.MP3),
                    text=text,
                    latency_ms=latency,
                    provider=self.provider_name,
                )

    async def synthesize_stream(self, text: str) -> AsyncIterator[AudioChunk]:
        """Stream synthesis using PlayHT streaming API."""
        url = f"{self.API_URL}/tts/stream"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-User-ID": self.user_id,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        voice_id = self.config.voice.voice_id if self.config.voice else self.config.voice_id

        payload = {
            "text": text,
            "voice": voice_id,
            "output_format": "mp3",
            "speed": self.config.speaking_rate,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                sequence = 0
                async for chunk in response.content.iter_any():
                    yield AudioChunk(
                        data=chunk,
                        format=AudioFormat(sample_rate=self.config.sample_rate, codec=AudioCodec.MP3),
                        sequence=sequence,
                    )
                    sequence += 1

    async def list_voices(self, language: str = "") -> List[TTSVoice]:
        """List available PlayHT voices."""
        url = f"{self.API_URL}/voices"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-User-ID": self.user_id,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                voices = []

                for voice in data:
                    voice_lang = voice.get("language_code", "en-US")

                    if language and not voice_lang.startswith(language[:2]):
                        continue

                    gender = VoiceGender.NEUTRAL
                    if voice.get("gender") == "male":
                        gender = VoiceGender.MALE
                    elif voice.get("gender") == "female":
                        gender = VoiceGender.FEMALE

                    voices.append(TTSVoice(
                        voice_id=voice["id"],
                        name=voice.get("name", voice["id"]),
                        language=voice_lang,
                        gender=gender,
                        provider=self.provider_name,
                        sample_url=voice.get("sample", ""),
                    ))

                return voices


class OpenAITTS(TTSProvider):
    """
    OpenAI TTS provider.

    Uses OpenAI's text-to-speech API.
    """

    API_URL = "https://api.openai.com/v1/audio/speech"

    VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def __init__(self, api_key: str, config: TTSConfig):
        super().__init__(api_key, config)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech using OpenAI TTS API."""
        cached = self._get_cached(text)
        if cached:
            return TTSResult(
                audio_data=cached,
                text=text,
                provider=self.provider_name,
            )

        voice = self.config.voice.voice_id if self.config.voice else self.config.voice_id
        if not voice or voice not in self.VOICES:
            voice = "alloy"

        model = self.config.model or "tts-1"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": "mp3",
            "speed": self.config.speaking_rate,
        }

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(self.API_URL, headers=headers, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"OpenAI TTS error: {error}")

                audio_data = await response.read()
                latency = (time.time() - start_time) * 1000

                self._set_cached(text, audio_data)

                return TTSResult(
                    audio_data=audio_data,
                    format=AudioFormat(sample_rate=24000, codec=AudioCodec.MP3),
                    text=text,
                    latency_ms=latency,
                    provider=self.provider_name,
                    characters=len(text),
                )

    async def synthesize_stream(self, text: str) -> AsyncIterator[AudioChunk]:
        """OpenAI TTS doesn't support streaming, use chunked synthesis."""
        result = await self.synthesize(text)
        yield result.to_audio_chunk()

    async def list_voices(self, language: str = "") -> List[TTSVoice]:
        """List available OpenAI voices."""
        voices = []

        voice_descriptions = {
            "alloy": {"gender": VoiceGender.NEUTRAL, "desc": "Neutral and balanced"},
            "echo": {"gender": VoiceGender.MALE, "desc": "Warm and reassuring"},
            "fable": {"gender": VoiceGender.NEUTRAL, "desc": "Expressive and dynamic"},
            "onyx": {"gender": VoiceGender.MALE, "desc": "Deep and authoritative"},
            "nova": {"gender": VoiceGender.FEMALE, "desc": "Warm and engaging"},
            "shimmer": {"gender": VoiceGender.FEMALE, "desc": "Clear and pleasant"},
        }

        for voice_id, info in voice_descriptions.items():
            voices.append(TTSVoice(
                voice_id=voice_id,
                name=voice_id.capitalize(),
                language="en-US",  # OpenAI TTS supports multilingual
                gender=info["gender"],
                provider=self.provider_name,
                metadata={"description": info["desc"]},
            ))

        return voices


class AzureTTS(TTSProvider):
    """
    Azure Cognitive Services TTS provider.

    Features:
    - 400+ neural voices
    - 140+ languages
    - SSML support
    - Custom neural voice
    """

    def __init__(self, api_key: str, config: TTSConfig, region: str = "eastus"):
        super().__init__(api_key, config)
        self.region = region
        self.endpoint = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        self.voices_endpoint = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list"

    @property
    def provider_name(self) -> str:
        return "azure"

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech using Azure TTS."""
        cached = self._get_cached(text)
        if cached:
            return TTSResult(
                audio_data=cached,
                text=text,
                provider=self.provider_name,
            )

        voice_name = self.config.voice.voice_id if self.config.voice else self.config.voice_id
        if not voice_name:
            voice_name = "en-US-JennyNeural"

        # Build SSML
        ssml = self._build_ssml(text, voice_name)

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-96kbitrate-mono-mp3",
        }

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, headers=headers, data=ssml) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Azure TTS error: {error}")

                audio_data = await response.read()
                latency = (time.time() - start_time) * 1000

                self._set_cached(text, audio_data)

                return TTSResult(
                    audio_data=audio_data,
                    format=AudioFormat(sample_rate=24000, codec=AudioCodec.MP3),
                    text=text,
                    latency_ms=latency,
                    provider=self.provider_name,
                    characters=len(text),
                )

    async def synthesize_stream(self, text: str) -> AsyncIterator[AudioChunk]:
        """Azure TTS with chunked response."""
        voice_name = self.config.voice.voice_id if self.config.voice else self.config.voice_id
        if not voice_name:
            voice_name = "en-US-JennyNeural"

        ssml = self._build_ssml(text, voice_name)

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-96kbitrate-mono-mp3",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, headers=headers, data=ssml) as response:
                sequence = 0
                async for chunk in response.content.iter_any():
                    yield AudioChunk(
                        data=chunk,
                        format=AudioFormat(sample_rate=24000, codec=AudioCodec.MP3),
                        sequence=sequence,
                    )
                    sequence += 1

    def _build_ssml(self, text: str, voice_name: str) -> str:
        """Build SSML for Azure TTS."""
        rate_percent = int((self.config.speaking_rate - 1.0) * 100)
        rate_str = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"

        pitch_str = f"+{int(self.config.pitch)}Hz" if self.config.pitch >= 0 else f"{int(self.config.pitch)}Hz"

        style = self.config.style.value if self.config.style != VoiceStyle.NEUTRAL else ""

        ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
            xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'>
            <voice name='{voice_name}'>
                <prosody rate='{rate_str}' pitch='{pitch_str}'>
        """

        if style:
            ssml += f"<mstts:express-as style='{style}' styledegree='{self.config.style_degree}'>"
            ssml += text
            ssml += "</mstts:express-as>"
        else:
            ssml += text

        ssml += """
                </prosody>
            </voice>
        </speak>"""

        return ssml

    async def list_voices(self, language: str = "") -> List[TTSVoice]:
        """List available Azure voices."""
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(self.voices_endpoint, headers=headers) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                voices = []

                for voice in data:
                    voice_lang = voice.get("Locale", "en-US")

                    if language and not voice_lang.startswith(language[:2]):
                        continue

                    gender = VoiceGender.NEUTRAL
                    if voice.get("Gender") == "Male":
                        gender = VoiceGender.MALE
                    elif voice.get("Gender") == "Female":
                        gender = VoiceGender.FEMALE

                    styles = []
                    for style in voice.get("StyleList", []):
                        try:
                            styles.append(VoiceStyle(style.lower()))
                        except ValueError:
                            pass

                    voices.append(TTSVoice(
                        voice_id=voice["ShortName"],
                        name=voice.get("DisplayName", voice["ShortName"]),
                        language=voice_lang,
                        gender=gender,
                        provider=self.provider_name,
                        sample_url=voice.get("SampleRateHertz", ""),
                        styles=styles,
                        metadata={
                            "voice_type": voice.get("VoiceType"),
                            "word_per_minute": voice.get("WordsPerMinute"),
                        },
                    ))

                return voices


class GoogleTTS(TTSProvider):
    """Google Cloud Text-to-Speech provider."""

    def __init__(self, api_key: str, config: TTSConfig):
        super().__init__(api_key, config)
        self._client = None

    @property
    def provider_name(self) -> str:
        return "google"

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech using Google Cloud TTS."""
        from google.cloud import texttospeech

        if not self._client:
            self._client = texttospeech.TextToSpeechAsyncClient()

        cached = self._get_cached(text)
        if cached:
            return TTSResult(
                audio_data=cached,
                text=text,
                provider=self.provider_name,
            )

        voice_name = self.config.voice.voice_id if self.config.voice else self.config.voice_id
        language = self.config.language or "en-US"

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language,
            name=voice_name if voice_name else None,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=self.config.speaking_rate,
            pitch=self.config.pitch,
            volume_gain_db=self.config.volume,
        )

        start_time = time.time()

        response = await self._client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        latency = (time.time() - start_time) * 1000

        self._set_cached(text, response.audio_content)

        return TTSResult(
            audio_data=response.audio_content,
            format=AudioFormat(sample_rate=24000, codec=AudioCodec.MP3),
            text=text,
            latency_ms=latency,
            provider=self.provider_name,
            characters=len(text),
        )

    async def synthesize_stream(self, text: str) -> AsyncIterator[AudioChunk]:
        """Google TTS doesn't support streaming, use single synthesis."""
        result = await self.synthesize(text)
        yield result.to_audio_chunk()

    async def list_voices(self, language: str = "") -> List[TTSVoice]:
        """List available Google Cloud voices."""
        from google.cloud import texttospeech

        if not self._client:
            self._client = texttospeech.TextToSpeechAsyncClient()

        response = await self._client.list_voices(language_code=language if language else None)

        voices = []
        for voice in response.voices:
            gender = VoiceGender.NEUTRAL
            if voice.ssml_gender == texttospeech.SsmlVoiceGender.MALE:
                gender = VoiceGender.MALE
            elif voice.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE:
                gender = VoiceGender.FEMALE

            for lang_code in voice.language_codes:
                voices.append(TTSVoice(
                    voice_id=voice.name,
                    name=voice.name,
                    language=lang_code,
                    gender=gender,
                    provider=self.provider_name,
                    metadata={
                        "natural_sample_rate": voice.natural_sample_rate_hertz,
                    },
                ))

        return voices


class TTSProviderFactory:
    """Factory for creating TTS provider instances."""

    _providers: Dict[str, type] = {
        "elevenlabs": ElevenLabsTTS,
        "playht": PlayHTTTS,
        "openai": OpenAITTS,
        "azure": AzureTTS,
        "google": GoogleTTS,
    }

    @classmethod
    def create(
        cls,
        provider_type: str,
        api_key: str,
        config: Optional[TTSConfig] = None,
        **kwargs,
    ) -> TTSProvider:
        """
        Create a TTS provider instance.

        Args:
            provider_type: Type of provider ('elevenlabs', 'openai', etc.)
            api_key: API key for the provider
            config: TTS configuration
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured TTSProvider instance
        """
        config = config or TTSConfig()

        provider_class = cls._providers.get(provider_type.lower())
        if not provider_class:
            raise ValueError(f"Unknown TTS provider: {provider_type}")

        return provider_class(api_key, config, **kwargs)

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """Register a custom TTS provider."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def available_providers(cls) -> List[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())

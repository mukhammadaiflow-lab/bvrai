"""
BVRAI Voice Engine - STT Provider Implementations

Multi-provider Speech-to-Text support with automatic failover,
streaming capabilities, and comprehensive error handling.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from .base import (
    AudioConfig,
    AudioFormat,
    InvalidAudioError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    STTConfig,
    STTError,
    STTProvider,
    STTProviderInterface,
    TranscriptionResult,
    TranscriptionStatus,
    TranscriptionWord,
    UnsupportedLanguageError,
    generate_request_id,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Deepgram Provider
# =============================================================================

class DeepgramSTTProvider(STTProviderInterface):
    """
    Deepgram Speech-to-Text provider.

    Supports real-time streaming with Nova-2 model.
    Excellent for low-latency conversational AI.
    """

    # Supported languages
    SUPPORTED_LANGUAGES = {
        "en", "en-US", "en-GB", "en-AU", "en-IN",
        "es", "es-ES", "es-419",
        "fr", "fr-FR", "fr-CA",
        "de", "de-DE",
        "it", "it-IT",
        "pt", "pt-BR", "pt-PT",
        "nl", "nl-NL",
        "hi", "hi-IN",
        "ja", "ja-JP",
        "ko", "ko-KR",
        "zh", "zh-CN", "zh-TW",
        "ru", "ru-RU",
        "pl", "pl-PL",
        "tr", "tr-TR",
        "uk", "uk-UA",
        "id", "id-ID",
        "ms", "ms-MY",
        "th", "th-TH",
        "vi", "vi-VN",
    }

    # Model pricing per minute of audio
    MODEL_PRICING = {
        "nova-2": 0.0043,
        "nova-2-general": 0.0043,
        "nova-2-meeting": 0.0043,
        "nova-2-phonecall": 0.0043,
        "nova-2-conversationalai": 0.0043,
        "nova": 0.0059,
        "enhanced": 0.0145,
        "base": 0.0125,
        "whisper-large": 0.0048,
        "whisper-medium": 0.0042,
        "whisper-small": 0.0038,
        "whisper-base": 0.0032,
        "whisper-tiny": 0.0030,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepgram.com/v1",
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("Deepgram API key is required")

        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_name(self) -> STTProvider:
        return STTProvider.DEEPGRAM

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "audio/wav",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def transcribe(
        self,
        audio_data: bytes,
        config: STTConfig
    ) -> TranscriptionResult:
        """Transcribe audio data using Deepgram."""
        request_id = generate_request_id()
        start_time = time.monotonic()

        # Validate input
        if not audio_data:
            raise InvalidAudioError("Empty audio data provided", provider="deepgram")

        if len(audio_data) < 100:
            raise InvalidAudioError(
                "Audio data too small (minimum 100 bytes)",
                provider="deepgram"
            )

        # Validate language
        lang = config.language.split("-")[0]
        if lang not in self.SUPPORTED_LANGUAGES and config.language not in self.SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Language '{config.language}' not supported by Deepgram",
                language=config.language,
                provider="deepgram"
            )

        try:
            client = await self._get_client()

            # Build query parameters
            params = {
                "model": config.model or "nova-2",
                "language": config.language,
                "punctuate": str(config.punctuate).lower(),
                "profanity_filter": str(config.profanity_filter).lower(),
                "diarize": str(config.diarize).lower(),
                "smart_format": str(config.smart_format).lower(),
            }

            # Add keywords if specified
            if config.keywords:
                params["keywords"] = ",".join(config.keywords)

            # Add custom vocabulary
            if config.custom_vocabulary:
                params["search"] = ",".join(config.custom_vocabulary)

            # Determine content type based on audio format
            content_type = self._get_content_type(config.audio_config)

            response = await client.post(
                "/listen",
                content=audio_data,
                params=params,
                headers={"Content-Type": content_type},
            )

            latency_ms = (time.monotonic() - start_time) * 1000

            if response.status_code == 429:
                raise ProviderRateLimitError(
                    "Deepgram rate limit exceeded",
                    provider="deepgram",
                    retry_after=int(response.headers.get("Retry-After", 60))
                )

            if response.status_code != 200:
                error_body = response.text
                raise STTError(
                    f"Deepgram API error: {response.status_code} - {error_body}",
                    provider="deepgram",
                    details={"status_code": response.status_code, "body": error_body}
                )

            # Parse response
            data = response.json()
            result = self._parse_response(data, config, request_id, latency_ms)

            # Calculate cost
            duration_minutes = result.duration / 60
            price_per_minute = self.MODEL_PRICING.get(config.model, 0.0043)
            result.processing_time_ms = latency_ms

            logger.info(
                f"Deepgram transcription completed",
                extra={
                    "request_id": request_id,
                    "duration": result.duration,
                    "latency_ms": latency_ms,
                    "text_length": len(result.text),
                }
            )

            return result

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                f"Deepgram request timed out after {self.timeout}s",
                provider="deepgram",
                details={"timeout": self.timeout}
            )
        except httpx.ConnectError as e:
            raise ProviderConnectionError(
                "Failed to connect to Deepgram API",
                provider="deepgram",
                details={"error": str(e)}
            )
        except (ProviderRateLimitError, ProviderTimeoutError, STTError):
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in Deepgram transcription: {e}")
            raise STTError(
                f"Unexpected error during transcription: {str(e)}",
                provider="deepgram",
                details={"error_type": type(e).__name__}
            )

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        config: STTConfig
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Stream transcription using Deepgram's WebSocket API.

        Note: This is a simplified implementation. For production,
        use the actual WebSocket streaming endpoint.
        """
        import websockets

        request_id = generate_request_id()

        # Build WebSocket URL
        ws_url = "wss://api.deepgram.com/v1/listen"
        params = {
            "model": config.model or "nova-2",
            "language": config.language,
            "punctuate": str(config.punctuate).lower(),
            "interim_results": str(config.interim_results).lower(),
            "endpointing": str(config.endpointing),
            "smart_format": str(config.smart_format).lower(),
        }
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{ws_url}?{query_string}"

        headers = {"Authorization": f"Token {self.api_key}"}

        try:
            async with websockets.connect(full_url, extra_headers=headers) as ws:
                # Start tasks for sending audio and receiving results
                send_task = asyncio.create_task(
                    self._send_audio_stream(ws, audio_stream)
                )

                try:
                    async for message in ws:
                        data = json.loads(message)

                        if "channel" in data:
                            result = self._parse_streaming_response(
                                data, config, request_id
                            )
                            if result.text.strip():
                                yield result

                            # Check if this is the final result
                            if data.get("is_final", False):
                                break

                except websockets.ConnectionClosed:
                    pass
                finally:
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass

        except Exception as e:
            logger.exception(f"Error in Deepgram streaming: {e}")
            raise STTError(
                f"Streaming transcription failed: {str(e)}",
                provider="deepgram"
            )

    async def _send_audio_stream(
        self,
        ws,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> None:
        """Send audio chunks to WebSocket."""
        try:
            async for chunk in audio_stream:
                await ws.send(chunk)
            # Send close message
            await ws.send(json.dumps({"type": "CloseStream"}))
        except Exception as e:
            logger.error(f"Error sending audio stream: {e}")

    def _parse_response(
        self,
        data: Dict[str, Any],
        config: STTConfig,
        request_id: str,
        latency_ms: float
    ) -> TranscriptionResult:
        """Parse Deepgram API response."""
        try:
            channels = data.get("results", {}).get("channels", [])
            if not channels:
                return TranscriptionResult(
                    id=request_id,
                    text="",
                    status=TranscriptionStatus.COMPLETED,
                    is_final=True,
                    provider=STTProvider.DEEPGRAM,
                    model=config.model,
                    latency_ms=latency_ms,
                )

            alternatives = channels[0].get("alternatives", [])
            if not alternatives:
                return TranscriptionResult(
                    id=request_id,
                    text="",
                    status=TranscriptionStatus.COMPLETED,
                    is_final=True,
                    provider=STTProvider.DEEPGRAM,
                    model=config.model,
                    latency_ms=latency_ms,
                )

            best = alternatives[0]
            transcript = best.get("transcript", "")
            confidence = best.get("confidence", 0.0)

            # Parse words with timestamps
            words = []
            for w in best.get("words", []):
                words.append(TranscriptionWord(
                    word=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    confidence=w.get("confidence", 0.0),
                    speaker=w.get("speaker"),
                ))

            # Get metadata
            metadata = data.get("metadata", {})
            duration = metadata.get("duration", 0.0)

            return TranscriptionResult(
                id=request_id,
                text=transcript,
                words=words,
                status=TranscriptionStatus.COMPLETED,
                is_final=True,
                confidence=confidence,
                language=config.language,
                duration=duration,
                provider=STTProvider.DEEPGRAM,
                model=config.model,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Error parsing Deepgram response: {e}")
            raise STTError(
                f"Failed to parse Deepgram response: {str(e)}",
                provider="deepgram"
            )

    def _parse_streaming_response(
        self,
        data: Dict[str, Any],
        config: STTConfig,
        request_id: str
    ) -> TranscriptionResult:
        """Parse streaming WebSocket response."""
        channel = data.get("channel", {})
        alternatives = channel.get("alternatives", [])

        if not alternatives:
            return TranscriptionResult(
                id=request_id,
                text="",
                status=TranscriptionStatus.PARTIAL,
                is_final=False,
                provider=STTProvider.DEEPGRAM,
                model=config.model,
            )

        best = alternatives[0]
        is_final = data.get("is_final", False)

        return TranscriptionResult(
            id=request_id,
            text=best.get("transcript", ""),
            confidence=best.get("confidence", 0.0),
            status=TranscriptionStatus.COMPLETED if is_final else TranscriptionStatus.PARTIAL,
            is_final=is_final,
            provider=STTProvider.DEEPGRAM,
            model=config.model,
        )

    def _get_content_type(self, audio_config: AudioConfig) -> str:
        """Get content type header for audio format."""
        format_map = {
            AudioFormat.PCM_16: "audio/l16",
            AudioFormat.WAV: "audio/wav",
            AudioFormat.MP3: "audio/mp3",
            AudioFormat.OGG_OPUS: "audio/ogg",
            AudioFormat.FLAC: "audio/flac",
            AudioFormat.WEBM: "audio/webm",
            AudioFormat.MULAW: "audio/basic",
        }
        return format_map.get(audio_config.format, "audio/wav")

    async def health_check(self) -> bool:
        """Check Deepgram API health."""
        try:
            client = await self._get_client()
            response = await client.get("/projects", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# =============================================================================
# AssemblyAI Provider
# =============================================================================

class AssemblyAISTTProvider(STTProviderInterface):
    """
    AssemblyAI Speech-to-Text provider.

    Known for high accuracy and advanced features like
    speaker diarization and content safety detection.
    """

    SUPPORTED_LANGUAGES = {
        "en", "en_us", "en_uk", "en_au",
        "es", "fr", "de", "it", "pt", "nl",
        "hi", "ja", "zh", "ko", "pl", "ru",
        "tr", "uk", "vi", "fi", "sv", "da",
        "no", "cs", "el", "he", "hu", "id",
        "ms", "ro", "sk", "th",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.assemblyai.com/v2",
        timeout: float = 120.0,  # Longer timeout for processing
    ):
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("AssemblyAI API key is required")

        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_name(self) -> STTProvider:
        return STTProvider.ASSEMBLYAI

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": self.api_key},
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def transcribe(
        self,
        audio_data: bytes,
        config: STTConfig
    ) -> TranscriptionResult:
        """Transcribe audio using AssemblyAI."""
        request_id = generate_request_id()
        start_time = time.monotonic()

        if not audio_data:
            raise InvalidAudioError("Empty audio data provided", provider="assemblyai")

        try:
            client = await self._get_client()

            # Step 1: Upload audio
            upload_response = await client.post(
                "/upload",
                content=audio_data,
                headers={"Content-Type": "application/octet-stream"},
            )

            if upload_response.status_code != 200:
                raise STTError(
                    f"Failed to upload audio: {upload_response.text}",
                    provider="assemblyai"
                )

            upload_url = upload_response.json().get("upload_url")

            # Step 2: Create transcription
            transcription_config = {
                "audio_url": upload_url,
                "language_code": config.language.replace("-", "_").lower(),
                "punctuate": config.punctuate,
                "format_text": config.smart_format,
            }

            if config.diarize:
                transcription_config["speaker_labels"] = True

            create_response = await client.post(
                "/transcript",
                json=transcription_config,
            )

            if create_response.status_code != 200:
                raise STTError(
                    f"Failed to create transcription: {create_response.text}",
                    provider="assemblyai"
                )

            transcript_id = create_response.json().get("id")

            # Step 3: Poll for completion
            while True:
                status_response = await client.get(f"/transcript/{transcript_id}")
                status_data = status_response.json()
                status = status_data.get("status")

                if status == "completed":
                    break
                elif status == "error":
                    raise STTError(
                        f"Transcription failed: {status_data.get('error')}",
                        provider="assemblyai"
                    )

                await asyncio.sleep(1.0)

            latency_ms = (time.monotonic() - start_time) * 1000

            # Parse result
            return self._parse_response(status_data, config, request_id, latency_ms)

        except (STTError, InvalidAudioError):
            raise
        except Exception as e:
            logger.exception(f"AssemblyAI transcription error: {e}")
            raise STTError(
                f"Transcription failed: {str(e)}",
                provider="assemblyai"
            )

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        config: STTConfig
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Stream transcription using AssemblyAI real-time API."""
        import websockets

        request_id = generate_request_id()
        ws_url = "wss://api.assemblyai.com/v2/realtime/ws"

        params = {
            "sample_rate": config.audio_config.sample_rate,
            "encoding": "pcm_s16le",
        }
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{ws_url}?{query_string}"

        try:
            async with websockets.connect(
                full_url,
                extra_headers={"Authorization": self.api_key}
            ) as ws:
                # Receive session start
                session_msg = await ws.recv()
                session_data = json.loads(session_msg)

                if session_data.get("message_type") != "SessionBegins":
                    raise STTError(
                        "Failed to start AssemblyAI session",
                        provider="assemblyai"
                    )

                # Start audio sending task
                send_task = asyncio.create_task(
                    self._send_audio_assemblyai(ws, audio_stream)
                )

                try:
                    async for message in ws:
                        data = json.loads(message)
                        msg_type = data.get("message_type")

                        if msg_type in ("PartialTranscript", "FinalTranscript"):
                            result = TranscriptionResult(
                                id=request_id,
                                text=data.get("text", ""),
                                confidence=data.get("confidence", 0.0),
                                status=TranscriptionStatus.COMPLETED if msg_type == "FinalTranscript" else TranscriptionStatus.PARTIAL,
                                is_final=msg_type == "FinalTranscript",
                                provider=STTProvider.ASSEMBLYAI,
                            )
                            if result.text.strip():
                                yield result

                except websockets.ConnectionClosed:
                    pass
                finally:
                    send_task.cancel()

        except Exception as e:
            logger.exception(f"AssemblyAI streaming error: {e}")
            raise STTError(
                f"Streaming failed: {str(e)}",
                provider="assemblyai"
            )

    async def _send_audio_assemblyai(
        self,
        ws,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> None:
        """Send audio to AssemblyAI WebSocket."""
        import base64
        try:
            async for chunk in audio_stream:
                # AssemblyAI expects base64-encoded audio
                encoded = base64.b64encode(chunk).decode("utf-8")
                await ws.send(json.dumps({"audio_data": encoded}))
            # Send termination message
            await ws.send(json.dumps({"terminate_session": True}))
        except Exception as e:
            logger.error(f"Error sending audio to AssemblyAI: {e}")

    def _parse_response(
        self,
        data: Dict[str, Any],
        config: STTConfig,
        request_id: str,
        latency_ms: float
    ) -> TranscriptionResult:
        """Parse AssemblyAI response."""
        words = []
        for w in data.get("words", []):
            words.append(TranscriptionWord(
                word=w.get("text", ""),
                start=w.get("start", 0) / 1000,  # Convert ms to seconds
                end=w.get("end", 0) / 1000,
                confidence=w.get("confidence", 0.0),
                speaker=w.get("speaker"),
            ))

        return TranscriptionResult(
            id=request_id,
            text=data.get("text", ""),
            words=words,
            status=TranscriptionStatus.COMPLETED,
            is_final=True,
            confidence=data.get("confidence", 0.0),
            language=config.language,
            duration=data.get("audio_duration", 0.0),
            provider=STTProvider.ASSEMBLYAI,
            latency_ms=latency_ms,
        )

    async def health_check(self) -> bool:
        """Check AssemblyAI API health."""
        try:
            client = await self._get_client()
            # Check by getting a small test
            response = await client.get("/", timeout=5.0)
            return response.status_code in (200, 404)  # 404 is expected for root
        except Exception:
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# =============================================================================
# OpenAI Whisper Provider
# =============================================================================

class OpenAIWhisperSTTProvider(STTProviderInterface):
    """
    OpenAI Whisper Speech-to-Text provider.

    Uses OpenAI's Whisper API for transcription.
    Great accuracy for many languages.
    """

    SUPPORTED_LANGUAGES = {
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
        "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
        "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
        "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
        "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
        "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
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
    def provider_name(self) -> STTProvider:
        return STTProvider.OPENAI_WHISPER

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def transcribe(
        self,
        audio_data: bytes,
        config: STTConfig
    ) -> TranscriptionResult:
        """Transcribe audio using OpenAI Whisper."""
        request_id = generate_request_id()
        start_time = time.monotonic()

        if not audio_data:
            raise InvalidAudioError("Empty audio data provided", provider="openai")

        try:
            client = await self._get_client()

            # Prepare multipart form data
            files = {
                "file": ("audio.wav", audio_data, "audio/wav"),
                "model": (None, config.model or "whisper-1"),
                "response_format": (None, "verbose_json"),
            }

            if config.language:
                lang_code = config.language.split("-")[0]
                files["language"] = (None, lang_code)

            response = await client.post(
                "/audio/transcriptions",
                files=files,
            )

            latency_ms = (time.monotonic() - start_time) * 1000

            if response.status_code == 429:
                raise ProviderRateLimitError(
                    "OpenAI rate limit exceeded",
                    provider="openai",
                    retry_after=int(response.headers.get("Retry-After", 60))
                )

            if response.status_code != 200:
                raise STTError(
                    f"OpenAI API error: {response.status_code} - {response.text}",
                    provider="openai"
                )

            data = response.json()

            # Parse words if available
            words = []
            if "words" in data:
                for w in data["words"]:
                    words.append(TranscriptionWord(
                        word=w.get("word", ""),
                        start=w.get("start", 0.0),
                        end=w.get("end", 0.0),
                        confidence=1.0,  # Whisper doesn't provide confidence
                    ))

            return TranscriptionResult(
                id=request_id,
                text=data.get("text", ""),
                words=words,
                status=TranscriptionStatus.COMPLETED,
                is_final=True,
                confidence=1.0,
                language=data.get("language", config.language),
                duration=data.get("duration", 0.0),
                provider=STTProvider.OPENAI_WHISPER,
                model=config.model or "whisper-1",
                latency_ms=latency_ms,
            )

        except (STTError, InvalidAudioError, ProviderRateLimitError):
            raise
        except Exception as e:
            logger.exception(f"OpenAI Whisper error: {e}")
            raise STTError(
                f"Transcription failed: {str(e)}",
                provider="openai"
            )

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        config: STTConfig
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        OpenAI Whisper doesn't support streaming.
        Buffer audio and transcribe when complete.
        """
        request_id = generate_request_id()

        # Collect all audio chunks
        audio_buffer = b""
        async for chunk in audio_stream:
            audio_buffer += chunk

        # Transcribe complete audio
        result = await self.transcribe(audio_buffer, config)
        result.id = request_id
        yield result

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
# Mock Provider (for testing)
# =============================================================================

class MockSTTProvider(STTProviderInterface):
    """Mock STT provider for testing."""

    def __init__(self, latency_ms: float = 100.0):
        self.latency_ms = latency_ms
        self._call_count = 0

    @property
    def provider_name(self) -> STTProvider:
        return STTProvider.MOCK

    async def transcribe(
        self,
        audio_data: bytes,
        config: STTConfig
    ) -> TranscriptionResult:
        """Return mock transcription."""
        await asyncio.sleep(self.latency_ms / 1000)
        self._call_count += 1

        return TranscriptionResult(
            id=generate_request_id(),
            text=f"Mock transcription #{self._call_count}. Audio size: {len(audio_data)} bytes.",
            status=TranscriptionStatus.COMPLETED,
            is_final=True,
            confidence=0.95,
            language=config.language,
            duration=len(audio_data) / 32000,  # Estimate
            provider=STTProvider.MOCK,
            model="mock",
            latency_ms=self.latency_ms,
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        config: STTConfig
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Stream mock transcriptions."""
        chunk_num = 0
        async for chunk in audio_stream:
            chunk_num += 1
            await asyncio.sleep(self.latency_ms / 1000)

            yield TranscriptionResult(
                id=generate_request_id(),
                text=f"Partial transcription chunk {chunk_num}",
                status=TranscriptionStatus.PARTIAL,
                is_final=False,
                provider=STTProvider.MOCK,
            )

        # Final result
        yield TranscriptionResult(
            id=generate_request_id(),
            text=f"Final transcription after {chunk_num} chunks",
            status=TranscriptionStatus.COMPLETED,
            is_final=True,
            provider=STTProvider.MOCK,
        )

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


# =============================================================================
# Provider Factory
# =============================================================================

def create_stt_provider(
    provider: STTProvider,
    **kwargs
) -> STTProviderInterface:
    """
    Factory function to create STT providers.

    Args:
        provider: The provider type to create
        **kwargs: Provider-specific configuration

    Returns:
        Configured STT provider instance
    """
    providers = {
        STTProvider.DEEPGRAM: DeepgramSTTProvider,
        STTProvider.ASSEMBLYAI: AssemblyAISTTProvider,
        STTProvider.OPENAI_WHISPER: OpenAIWhisperSTTProvider,
        STTProvider.MOCK: MockSTTProvider,
    }

    if provider not in providers:
        raise ValueError(f"Unsupported STT provider: {provider}")

    return providers[provider](**kwargs)

"""
Speech-to-Text (STT) module.

Provides a unified interface for multiple STT providers:
- Deepgram: Fast, accurate, real-time streaming
- OpenAI Whisper: High accuracy, multi-language
- Google Cloud Speech: Enterprise-grade, streaming
- Azure Speech Services: Enterprise-grade, streaming
- AssemblyAI: Fast, accurate, real-time

Features:
- Streaming and batch transcription
- Real-time interim results
- Word-level timestamps
- Speaker diarization support
- Language detection
- Custom vocabulary/keywords
"""

import asyncio
import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union
import aiohttp
import websockets

from .audio import AudioFormat, AudioChunk, AudioCodec

logger = logging.getLogger(__name__)


class STTProviderType(str, Enum):
    """Supported STT providers."""
    DEEPGRAM = "deepgram"
    WHISPER = "whisper"
    GOOGLE = "google"
    AZURE = "azure"
    ASSEMBLYAI = "assemblyai"


class TranscriptionType(str, Enum):
    """Types of transcription results."""
    INTERIM = "interim"       # Partial/in-progress result
    FINAL = "final"           # Final confirmed result
    CORRECTION = "correction" # Correction of previous result


@dataclass
class WordInfo:
    """Information about a single word in transcription."""
    word: str
    start_ms: float
    end_ms: float
    confidence: float = 0.0
    speaker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": self.confidence,
            "speaker": self.speaker,
        }


@dataclass
class TranscriptionSegment:
    """
    A segment of transcribed speech.

    Attributes:
        text: Transcribed text
        start_ms: Start timestamp in milliseconds
        end_ms: End timestamp in milliseconds
        confidence: Transcription confidence (0.0 - 1.0)
        words: Word-level information
        language: Detected language code
        speaker: Speaker identifier (if diarization enabled)
        is_final: Whether this is a final result
    """
    text: str
    start_ms: float = 0.0
    end_ms: float = 0.0
    confidence: float = 0.0
    words: List[WordInfo] = field(default_factory=list)
    language: str = ""
    speaker: Optional[str] = None
    is_final: bool = False
    segment_id: str = ""

    @property
    def duration_ms(self) -> float:
        """Duration of this segment in milliseconds."""
        return self.end_ms - self.start_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": self.confidence,
            "words": [w.to_dict() for w in self.words],
            "language": self.language,
            "speaker": self.speaker,
            "is_final": self.is_final,
            "segment_id": self.segment_id,
        }


@dataclass
class STTResult:
    """
    Complete STT result containing all transcription data.

    Attributes:
        text: Full transcribed text
        segments: List of transcription segments
        language: Primary detected language
        duration_ms: Total audio duration processed
        transcription_type: Type of result (interim/final)
    """
    text: str = ""
    segments: List[TranscriptionSegment] = field(default_factory=list)
    language: str = ""
    duration_ms: float = 0.0
    transcription_type: TranscriptionType = TranscriptionType.FINAL

    # Metadata
    provider: str = ""
    latency_ms: float = 0.0
    audio_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration_ms": self.duration_ms,
            "transcription_type": self.transcription_type.value,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
        }


@dataclass
class STTConfig:
    """
    Configuration for STT providers.

    Attributes:
        language: Target language code (e.g., 'en-US')
        model: Model to use (provider-specific)
        sample_rate: Expected audio sample rate
        encoding: Audio encoding format

        interim_results: Enable interim/partial results
        punctuation: Enable automatic punctuation
        word_timestamps: Enable word-level timestamps
        diarization: Enable speaker diarization
        num_speakers: Expected number of speakers (for diarization)

        keywords: Custom keywords/vocabulary for boosting
        profanity_filter: Enable profanity filtering

        max_alternatives: Maximum number of alternative transcriptions
        timeout_seconds: Request timeout
    """
    language: str = "en-US"
    model: str = ""  # Provider-specific model
    sample_rate: int = 16000
    encoding: str = "linear16"

    interim_results: bool = True
    punctuation: bool = True
    word_timestamps: bool = True
    diarization: bool = False
    num_speakers: int = 2

    keywords: List[str] = field(default_factory=list)
    keyword_boost: float = 1.5
    profanity_filter: bool = False

    max_alternatives: int = 1
    timeout_seconds: float = 30.0

    # Provider-specific options
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "model": self.model,
            "sample_rate": self.sample_rate,
            "encoding": self.encoding,
            "interim_results": self.interim_results,
            "punctuation": self.punctuation,
            "word_timestamps": self.word_timestamps,
            "diarization": self.diarization,
            "num_speakers": self.num_speakers,
            "keywords": self.keywords,
            "profanity_filter": self.profanity_filter,
        }


class STTProvider(ABC):
    """Abstract base class for STT providers."""

    def __init__(self, api_key: str, config: STTConfig):
        self.api_key = api_key
        self.config = config
        self._is_connected = False
        self._callbacks: List[Callable[[STTResult], None]] = []

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of this STT provider."""
        pass

    @property
    def is_connected(self) -> bool:
        """Whether the provider is connected."""
        return self._is_connected

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the STT service."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the STT service."""
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[STTResult]:
        """
        Transcribe streaming audio.

        Args:
            audio_stream: Async iterator of audio chunks

        Yields:
            STTResult objects as transcription becomes available
        """
        pass

    @abstractmethod
    async def transcribe(self, audio_data: bytes, format: AudioFormat) -> STTResult:
        """
        Transcribe a complete audio file.

        Args:
            audio_data: Complete audio data
            format: Audio format specification

        Returns:
            Complete STTResult
        """
        pass

    def add_callback(self, callback: Callable[[STTResult], None]) -> None:
        """Add callback for transcription results."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[STTResult], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def _emit_result(self, result: STTResult) -> None:
        """Emit result to all callbacks."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Error in STT callback: {e}")


class DeepgramSTT(STTProvider):
    """
    Deepgram STT provider.

    Features:
    - Ultra-low latency streaming
    - High accuracy
    - Real-time interim results
    - Speaker diarization
    - Custom vocabulary
    """

    STREAMING_URL = "wss://api.deepgram.com/v1/listen"
    BATCH_URL = "https://api.deepgram.com/v1/listen"

    def __init__(self, api_key: str, config: STTConfig):
        super().__init__(api_key, config)
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._result_queue: asyncio.Queue = asyncio.Queue()

    @property
    def provider_name(self) -> str:
        return "deepgram"

    async def connect(self) -> None:
        """Connect to Deepgram streaming API."""
        params = self._build_params()
        url = f"{self.STREAMING_URL}?{params}"

        headers = {
            "Authorization": f"Token {self.api_key}",
        }

        try:
            self._ws = await websockets.connect(
                url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            self._is_connected = True
            logger.info("Connected to Deepgram STT")

            # Start receive task
            self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Deepgram."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._is_connected = False
        logger.info("Disconnected from Deepgram STT")

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[STTResult]:
        """Stream audio and yield transcription results."""
        if not self._is_connected:
            await self.connect()

        # Send audio in background
        send_task = asyncio.create_task(self._send_audio(audio_stream))

        try:
            while True:
                try:
                    result = await asyncio.wait_for(
                        self._result_queue.get(),
                        timeout=self.config.timeout_seconds,
                    )
                    if result is None:
                        break
                    yield result
                except asyncio.TimeoutError:
                    break

        finally:
            send_task.cancel()

    async def transcribe(self, audio_data: bytes, format: AudioFormat) -> STTResult:
        """Transcribe complete audio using batch API."""
        params = self._build_params()
        url = f"{self.BATCH_URL}?{params}"

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": self._get_content_type(format),
        }

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=audio_data) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Deepgram error: {error}")

                data = await response.json()
                latency = (time.time() - start_time) * 1000

                return self._parse_response(data, latency)

    async def _send_audio(self, audio_stream: AsyncIterator[AudioChunk]) -> None:
        """Send audio chunks to Deepgram."""
        try:
            async for chunk in audio_stream:
                if self._ws and self._is_connected:
                    await self._ws.send(chunk.data)

            # Send close message
            if self._ws and self._is_connected:
                await self._ws.send(json.dumps({"type": "CloseStream"}))

        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {e}")

    async def _receive_loop(self) -> None:
        """Receive and process results from Deepgram."""
        try:
            while self._ws and self._is_connected:
                message = await self._ws.recv()

                try:
                    data = json.loads(message)
                    result = self._parse_streaming_response(data)
                    if result:
                        await self._result_queue.put(result)
                        await self._emit_result(result)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Deepgram: {message}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Deepgram connection closed")
        except Exception as e:
            logger.error(f"Error receiving from Deepgram: {e}")
        finally:
            await self._result_queue.put(None)

    def _build_params(self) -> str:
        """Build query parameters for Deepgram API."""
        params = {
            "language": self.config.language,
            "encoding": self._map_encoding(),
            "sample_rate": self.config.sample_rate,
            "channels": 1,
            "punctuate": str(self.config.punctuation).lower(),
            "interim_results": str(self.config.interim_results).lower(),
        }

        if self.config.model:
            params["model"] = self.config.model

        if self.config.diarization:
            params["diarize"] = "true"
            params["diarize_version"] = "2"

        if self.config.word_timestamps:
            params["utterances"] = "true"

        if self.config.keywords:
            params["keywords"] = ",".join(self.config.keywords)

        if self.config.profanity_filter:
            params["profanity_filter"] = "true"

        # Add extra options
        params.update(self.config.extra_options)

        return "&".join(f"{k}={v}" for k, v in params.items())

    def _map_encoding(self) -> str:
        """Map audio format to Deepgram encoding."""
        encoding_map = {
            "linear16": "linear16",
            "pcm": "linear16",
            "mulaw": "mulaw",
            "alaw": "alaw",
            "opus": "opus",
            "flac": "flac",
            "mp3": "mp3",
        }
        return encoding_map.get(self.config.encoding, "linear16")

    def _get_content_type(self, format: AudioFormat) -> str:
        """Get content type for batch API."""
        content_type_map = {
            AudioCodec.PCM: "audio/raw",
            AudioCodec.WAV: "audio/wav",
            AudioCodec.MULAW: "audio/mulaw",
            AudioCodec.MP3: "audio/mpeg",
            AudioCodec.FLAC: "audio/flac",
        }
        return content_type_map.get(format.codec, "audio/raw")

    def _parse_streaming_response(self, data: Dict) -> Optional[STTResult]:
        """Parse streaming response from Deepgram."""
        if data.get("type") != "Results":
            return None

        channel = data.get("channel", {})
        alternatives = channel.get("alternatives", [])

        if not alternatives:
            return None

        alt = alternatives[0]
        text = alt.get("transcript", "")

        if not text:
            return None

        is_final = data.get("is_final", False)

        # Parse words
        words = []
        for word_data in alt.get("words", []):
            words.append(WordInfo(
                word=word_data.get("word", ""),
                start_ms=word_data.get("start", 0) * 1000,
                end_ms=word_data.get("end", 0) * 1000,
                confidence=word_data.get("confidence", 0),
                speaker=word_data.get("speaker"),
            ))

        segment = TranscriptionSegment(
            text=text,
            start_ms=data.get("start", 0) * 1000,
            end_ms=(data.get("start", 0) + data.get("duration", 0)) * 1000,
            confidence=alt.get("confidence", 0),
            words=words,
            is_final=is_final,
        )

        return STTResult(
            text=text,
            segments=[segment],
            language=data.get("metadata", {}).get("detected_language", self.config.language),
            transcription_type=TranscriptionType.FINAL if is_final else TranscriptionType.INTERIM,
            provider=self.provider_name,
        )

    def _parse_response(self, data: Dict, latency_ms: float) -> STTResult:
        """Parse batch response from Deepgram."""
        results = data.get("results", {})
        channels = results.get("channels", [{}])
        alternatives = channels[0].get("alternatives", [{}]) if channels else [{}]
        alt = alternatives[0] if alternatives else {}

        text = alt.get("transcript", "")
        confidence = alt.get("confidence", 0)

        # Parse words
        words = []
        for word_data in alt.get("words", []):
            words.append(WordInfo(
                word=word_data.get("word", ""),
                start_ms=word_data.get("start", 0) * 1000,
                end_ms=word_data.get("end", 0) * 1000,
                confidence=word_data.get("confidence", 0),
                speaker=word_data.get("speaker"),
            ))

        segment = TranscriptionSegment(
            text=text,
            confidence=confidence,
            words=words,
            is_final=True,
        )

        return STTResult(
            text=text,
            segments=[segment],
            language=results.get("metadata", {}).get("detected_language", self.config.language),
            duration_ms=results.get("metadata", {}).get("duration", 0) * 1000,
            transcription_type=TranscriptionType.FINAL,
            provider=self.provider_name,
            latency_ms=latency_ms,
        )


class WhisperSTT(STTProvider):
    """
    OpenAI Whisper STT provider.

    Uses OpenAI's Whisper API for high-accuracy transcription.
    Supports batch transcription and translation.
    """

    API_URL = "https://api.openai.com/v1/audio/transcriptions"

    def __init__(self, api_key: str, config: STTConfig):
        super().__init__(api_key, config)

    @property
    def provider_name(self) -> str:
        return "whisper"

    async def connect(self) -> None:
        """Whisper uses batch API, no persistent connection needed."""
        self._is_connected = True

    async def disconnect(self) -> None:
        """No connection to close."""
        self._is_connected = False

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[STTResult]:
        """
        Stream transcription using chunked approach.

        Note: Whisper API doesn't support true streaming,
        so we accumulate chunks and transcribe periodically.
        """
        buffer = bytearray()
        chunk_size = self.config.sample_rate * 2 * 5  # 5 seconds

        async for chunk in audio_stream:
            buffer.extend(chunk.data)

            if len(buffer) >= chunk_size:
                result = await self.transcribe(
                    bytes(buffer),
                    AudioFormat.wideband(),
                )
                yield result
                buffer.clear()

        # Transcribe remaining audio
        if buffer:
            result = await self.transcribe(bytes(buffer), AudioFormat.wideband())
            yield result

    async def transcribe(self, audio_data: bytes, format: AudioFormat) -> STTResult:
        """Transcribe audio using Whisper API."""
        import io

        # Convert to WAV format for API
        from .audio import WAVEncoder
        wav_encoder = WAVEncoder()
        wav_data = wav_encoder.encode(audio_data, format)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        # Build form data
        form = aiohttp.FormData()
        form.add_field(
            "file",
            io.BytesIO(wav_data),
            filename="audio.wav",
            content_type="audio/wav",
        )
        form.add_field("model", self.config.model or "whisper-1")
        form.add_field("language", self.config.language[:2])  # Whisper uses 2-letter codes
        form.add_field("response_format", "verbose_json")

        if self.config.word_timestamps:
            form.add_field("timestamp_granularities[]", "word")

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.API_URL,
                headers=headers,
                data=form,
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Whisper error: {error}")

                data = await response.json()
                latency = (time.time() - start_time) * 1000

                return self._parse_response(data, latency)

    def _parse_response(self, data: Dict, latency_ms: float) -> STTResult:
        """Parse Whisper API response."""
        text = data.get("text", "")

        segments = []
        for seg in data.get("segments", []):
            words = []
            for word_data in seg.get("words", []):
                words.append(WordInfo(
                    word=word_data.get("word", ""),
                    start_ms=word_data.get("start", 0) * 1000,
                    end_ms=word_data.get("end", 0) * 1000,
                ))

            segments.append(TranscriptionSegment(
                text=seg.get("text", ""),
                start_ms=seg.get("start", 0) * 1000,
                end_ms=seg.get("end", 0) * 1000,
                words=words,
                is_final=True,
            ))

        return STTResult(
            text=text,
            segments=segments,
            language=data.get("language", self.config.language),
            duration_ms=data.get("duration", 0) * 1000,
            transcription_type=TranscriptionType.FINAL,
            provider=self.provider_name,
            latency_ms=latency_ms,
        )


class GoogleSTT(STTProvider):
    """Google Cloud Speech-to-Text provider."""

    def __init__(self, api_key: str, config: STTConfig):
        super().__init__(api_key, config)
        self._client = None

    @property
    def provider_name(self) -> str:
        return "google"

    async def connect(self) -> None:
        """Initialize Google Cloud client."""
        try:
            from google.cloud import speech
            self._client = speech.SpeechAsyncClient()
            self._is_connected = True
            logger.info("Connected to Google Cloud Speech")
        except ImportError:
            raise ImportError("google-cloud-speech package required")

    async def disconnect(self) -> None:
        """Close Google Cloud client."""
        self._client = None
        self._is_connected = False

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[STTResult]:
        """Stream audio to Google Cloud Speech."""
        from google.cloud import speech

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.config.sample_rate,
            language_code=self.config.language,
            enable_automatic_punctuation=self.config.punctuation,
            enable_word_time_offsets=self.config.word_timestamps,
            diarization_config=speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=self.config.diarization,
                min_speaker_count=2,
                max_speaker_count=self.config.num_speakers,
            ) if self.config.diarization else None,
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=self.config.interim_results,
        )

        async def request_generator():
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            async for chunk in audio_stream:
                yield speech.StreamingRecognizeRequest(audio_content=chunk.data)

        responses = await self._client.streaming_recognize(request_generator())

        async for response in responses:
            for result in response.results:
                yield self._parse_result(result)

    async def transcribe(self, audio_data: bytes, format: AudioFormat) -> STTResult:
        """Transcribe using Google Cloud batch API."""
        from google.cloud import speech

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=format.sample_rate,
            language_code=self.config.language,
            enable_automatic_punctuation=self.config.punctuation,
            enable_word_time_offsets=self.config.word_timestamps,
        )

        audio = speech.RecognitionAudio(content=audio_data)

        start_time = time.time()
        response = await self._client.recognize(config=config, audio=audio)
        latency = (time.time() - start_time) * 1000

        # Combine all results
        text_parts = []
        all_segments = []

        for result in response.results:
            segment = self._parse_result(result)
            text_parts.append(segment.segments[0].text if segment.segments else "")
            all_segments.extend(segment.segments)

        return STTResult(
            text=" ".join(text_parts),
            segments=all_segments,
            language=self.config.language,
            transcription_type=TranscriptionType.FINAL,
            provider=self.provider_name,
            latency_ms=latency,
        )

    def _parse_result(self, result) -> STTResult:
        """Parse Google Cloud Speech result."""
        if not result.alternatives:
            return STTResult(provider=self.provider_name)

        alt = result.alternatives[0]
        text = alt.transcript
        confidence = alt.confidence

        words = []
        for word_info in alt.words:
            words.append(WordInfo(
                word=word_info.word,
                start_ms=word_info.start_time.total_seconds() * 1000,
                end_ms=word_info.end_time.total_seconds() * 1000,
                speaker=str(word_info.speaker_tag) if hasattr(word_info, 'speaker_tag') else None,
            ))

        segment = TranscriptionSegment(
            text=text,
            confidence=confidence,
            words=words,
            is_final=result.is_final,
        )

        return STTResult(
            text=text,
            segments=[segment],
            transcription_type=TranscriptionType.FINAL if result.is_final else TranscriptionType.INTERIM,
            provider=self.provider_name,
        )


class AzureSTT(STTProvider):
    """Azure Cognitive Services Speech-to-Text provider."""

    def __init__(self, api_key: str, config: STTConfig, region: str = "eastus"):
        super().__init__(api_key, config)
        self.region = region
        self._speech_config = None
        self._recognizer = None

    @property
    def provider_name(self) -> str:
        return "azure"

    async def connect(self) -> None:
        """Initialize Azure Speech SDK."""
        try:
            import azure.cognitiveservices.speech as speechsdk

            self._speech_config = speechsdk.SpeechConfig(
                subscription=self.api_key,
                region=self.region,
            )
            self._speech_config.speech_recognition_language = self.config.language

            if self.config.word_timestamps:
                self._speech_config.request_word_level_timestamps()

            self._is_connected = True
            logger.info("Connected to Azure Speech Services")

        except ImportError:
            raise ImportError("azure-cognitiveservices-speech package required")

    async def disconnect(self) -> None:
        """Close Azure Speech connection."""
        self._recognizer = None
        self._is_connected = False

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[STTResult]:
        """Stream audio to Azure Speech."""
        import azure.cognitiveservices.speech as speechsdk

        # Create push stream for audio
        push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._speech_config,
            audio_config=audio_config,
        )

        result_queue = asyncio.Queue()

        def handle_result(evt):
            asyncio.create_task(result_queue.put(evt))

        recognizer.recognizing.connect(handle_result)
        recognizer.recognized.connect(handle_result)
        recognizer.session_stopped.connect(lambda e: asyncio.create_task(result_queue.put(None)))

        recognizer.start_continuous_recognition()

        # Send audio in background
        async def send_audio():
            async for chunk in audio_stream:
                push_stream.write(chunk.data)
            push_stream.close()

        send_task = asyncio.create_task(send_audio())

        try:
            while True:
                evt = await result_queue.get()
                if evt is None:
                    break

                result = self._parse_event(evt)
                if result:
                    yield result

        finally:
            recognizer.stop_continuous_recognition()
            send_task.cancel()

    async def transcribe(self, audio_data: bytes, format: AudioFormat) -> STTResult:
        """Transcribe using Azure batch API."""
        import azure.cognitiveservices.speech as speechsdk

        # Create audio config from bytes
        push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._speech_config,
            audio_config=audio_config,
        )

        push_stream.write(audio_data)
        push_stream.close()

        start_time = time.time()
        result = recognizer.recognize_once()
        latency = (time.time() - start_time) * 1000

        return self._parse_sdk_result(result, latency)

    def _parse_event(self, evt) -> Optional[STTResult]:
        """Parse Azure Speech event."""
        import azure.cognitiveservices.speech as speechsdk

        if hasattr(evt, 'result'):
            result = evt.result

            if result.reason == speechsdk.ResultReason.RecognizingSpeech:
                return STTResult(
                    text=result.text,
                    segments=[TranscriptionSegment(text=result.text, is_final=False)],
                    transcription_type=TranscriptionType.INTERIM,
                    provider=self.provider_name,
                )

            elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return self._parse_sdk_result(result, 0)

        return None

    def _parse_sdk_result(self, result, latency_ms: float) -> STTResult:
        """Parse Azure SDK result."""
        import azure.cognitiveservices.speech as speechsdk

        if result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return STTResult(provider=self.provider_name)

        text = result.text

        words = []
        if hasattr(result, 'best_json'):
            try:
                details = json.loads(result.best_json)
                for word_data in details.get("Words", []):
                    words.append(WordInfo(
                        word=word_data.get("Word", ""),
                        start_ms=word_data.get("Offset", 0) / 10000,
                        end_ms=(word_data.get("Offset", 0) + word_data.get("Duration", 0)) / 10000,
                        confidence=word_data.get("Confidence", 0),
                    ))
            except Exception:
                pass

        segment = TranscriptionSegment(
            text=text,
            words=words,
            is_final=True,
        )

        return STTResult(
            text=text,
            segments=[segment],
            transcription_type=TranscriptionType.FINAL,
            provider=self.provider_name,
            latency_ms=latency_ms,
        )


class AssemblyAISTT(STTProvider):
    """AssemblyAI Speech-to-Text provider."""

    STREAMING_URL = "wss://api.assemblyai.com/v2/realtime/ws"
    BATCH_URL = "https://api.assemblyai.com/v2"

    def __init__(self, api_key: str, config: STTConfig):
        super().__init__(api_key, config)
        self._ws = None
        self._session_id = None

    @property
    def provider_name(self) -> str:
        return "assemblyai"

    async def connect(self) -> None:
        """Connect to AssemblyAI real-time API."""
        url = f"{self.STREAMING_URL}?sample_rate={self.config.sample_rate}"

        headers = {
            "Authorization": self.api_key,
        }

        self._ws = await websockets.connect(url, extra_headers=headers)
        self._is_connected = True
        logger.info("Connected to AssemblyAI STT")

    async def disconnect(self) -> None:
        """Disconnect from AssemblyAI."""
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._is_connected = False

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[STTResult]:
        """Stream audio to AssemblyAI."""
        if not self._is_connected:
            await self.connect()

        async def send_audio():
            async for chunk in audio_stream:
                if self._ws:
                    await self._ws.send(json.dumps({
                        "audio_data": base64.b64encode(chunk.data).decode()
                    }))
            if self._ws:
                await self._ws.send(json.dumps({"terminate_session": True}))

        send_task = asyncio.create_task(send_audio())

        try:
            while self._ws:
                message = await self._ws.recv()
                data = json.loads(message)

                if data.get("message_type") == "FinalTranscript":
                    yield self._parse_response(data, is_final=True)
                elif data.get("message_type") == "PartialTranscript":
                    yield self._parse_response(data, is_final=False)
                elif data.get("message_type") == "SessionTerminated":
                    break

        finally:
            send_task.cancel()

    async def transcribe(self, audio_data: bytes, format: AudioFormat) -> STTResult:
        """Transcribe using AssemblyAI batch API."""
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }

        # Upload audio
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.BATCH_URL}/upload",
                headers={"Authorization": self.api_key},
                data=audio_data,
            ) as response:
                upload_data = await response.json()
                audio_url = upload_data.get("upload_url")

            # Request transcription
            transcript_request = {
                "audio_url": audio_url,
                "language_code": self.config.language[:2],
                "punctuate": self.config.punctuation,
                "word_boost": self.config.keywords,
            }

            if self.config.diarization:
                transcript_request["speaker_labels"] = True

            start_time = time.time()

            async with session.post(
                f"{self.BATCH_URL}/transcript",
                headers=headers,
                json=transcript_request,
            ) as response:
                data = await response.json()
                transcript_id = data.get("id")

            # Poll for completion
            while True:
                async with session.get(
                    f"{self.BATCH_URL}/transcript/{transcript_id}",
                    headers=headers,
                ) as response:
                    data = await response.json()
                    status = data.get("status")

                    if status == "completed":
                        latency = (time.time() - start_time) * 1000
                        return self._parse_batch_response(data, latency)
                    elif status == "error":
                        raise Exception(f"AssemblyAI error: {data.get('error')}")

                await asyncio.sleep(1)

    def _parse_response(self, data: Dict, is_final: bool) -> STTResult:
        """Parse streaming response."""
        text = data.get("text", "")

        words = []
        for word_data in data.get("words", []):
            words.append(WordInfo(
                word=word_data.get("text", ""),
                start_ms=word_data.get("start", 0),
                end_ms=word_data.get("end", 0),
                confidence=word_data.get("confidence", 0),
            ))

        segment = TranscriptionSegment(
            text=text,
            start_ms=data.get("audio_start", 0),
            end_ms=data.get("audio_end", 0),
            words=words,
            is_final=is_final,
        )

        return STTResult(
            text=text,
            segments=[segment],
            transcription_type=TranscriptionType.FINAL if is_final else TranscriptionType.INTERIM,
            provider=self.provider_name,
        )

    def _parse_batch_response(self, data: Dict, latency_ms: float) -> STTResult:
        """Parse batch API response."""
        text = data.get("text", "")

        words = []
        for word_data in data.get("words", []):
            words.append(WordInfo(
                word=word_data.get("text", ""),
                start_ms=word_data.get("start", 0),
                end_ms=word_data.get("end", 0),
                confidence=word_data.get("confidence", 0),
                speaker=word_data.get("speaker"),
            ))

        # Parse utterances as segments
        segments = []
        for utterance in data.get("utterances", [{"text": text}]):
            segments.append(TranscriptionSegment(
                text=utterance.get("text", ""),
                start_ms=utterance.get("start", 0),
                end_ms=utterance.get("end", 0),
                confidence=utterance.get("confidence", 0),
                speaker=utterance.get("speaker"),
                is_final=True,
            ))

        return STTResult(
            text=text,
            segments=segments,
            duration_ms=data.get("audio_duration", 0) * 1000,
            transcription_type=TranscriptionType.FINAL,
            provider=self.provider_name,
            latency_ms=latency_ms,
        )


class STTProviderFactory:
    """Factory for creating STT provider instances."""

    _providers: Dict[str, type] = {
        "deepgram": DeepgramSTT,
        "whisper": WhisperSTT,
        "google": GoogleSTT,
        "azure": AzureSTT,
        "assemblyai": AssemblyAISTT,
    }

    @classmethod
    def create(
        cls,
        provider_type: str,
        api_key: str,
        config: Optional[STTConfig] = None,
        **kwargs,
    ) -> STTProvider:
        """
        Create an STT provider instance.

        Args:
            provider_type: Type of provider ('deepgram', 'whisper', etc.)
            api_key: API key for the provider
            config: STT configuration
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured STTProvider instance
        """
        config = config or STTConfig()

        provider_class = cls._providers.get(provider_type.lower())
        if not provider_class:
            raise ValueError(f"Unknown STT provider: {provider_type}")

        return provider_class(api_key, config, **kwargs)

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """Register a custom STT provider."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def available_providers(cls) -> List[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())

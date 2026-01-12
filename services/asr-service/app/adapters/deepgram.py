"""Deepgram ASR adapter with real-time streaming support."""

import asyncio
import time
from typing import AsyncIterator, Optional

import structlog
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

from app.adapters.base import (
    ASRAdapter,
    TranscriptEvent,
    TranscriptResult,
    TranscriptType,
    VADEvent,
    Word,
)
from app.config import get_settings

logger = structlog.get_logger()


class DeepgramAdapter(ASRAdapter):
    """
    Deepgram ASR adapter using their real-time streaming API.

    Features:
    - Real-time streaming transcription
    - Interim (partial) results
    - Voice Activity Detection (VAD)
    - Endpointing (automatic speech boundary detection)
    - Multiple model options (nova-2, nova, enhanced, base)
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__()
        settings = get_settings()
        self.api_key = api_key or settings.deepgram_api_key
        self.settings = settings

        if not self.api_key:
            raise ValueError("Deepgram API key is required")

        # Create Deepgram client
        config = DeepgramClientOptions(
            verbose=settings.debug,
        )
        self.client = DeepgramClient(self.api_key, config)

        self._connection = None
        self._session_id: Optional[str] = None
        self._event_queue: asyncio.Queue[TranscriptEvent] = asyncio.Queue()
        self._is_connected = False
        self._last_transcript_time = 0.0

        self.logger = logger.bind(adapter="deepgram")

    @property
    def name(self) -> str:
        return "deepgram"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_interim_results(self) -> bool:
        return True

    async def connect(self, session_id: str, options: Optional[dict] = None) -> None:
        """Connect to Deepgram streaming API."""
        self._session_id = session_id
        self.logger = self.logger.bind(session_id=session_id)

        try:
            # Get live transcription options
            live_options = self._build_options(options)

            # Create live transcription connection
            self._connection = self.client.listen.asynclive.v("1")

            # Register event handlers
            self._connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self._connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            self._connection.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
            self._connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
            self._connection.on(LiveTranscriptionEvents.Metadata, self._on_metadata)
            self._connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self._connection.on(LiveTranscriptionEvents.Close, self._on_close)

            # Start the connection
            started = await self._connection.start(live_options)

            if not started:
                raise ConnectionError("Failed to start Deepgram connection")

            self._is_connected = True
            self.logger.info("Connected to Deepgram", model=self.settings.deepgram_model)

        except Exception as e:
            self.logger.error("Failed to connect to Deepgram", error=str(e))
            raise

    def _build_options(self, custom_options: Optional[dict] = None) -> LiveOptions:
        """Build Deepgram live transcription options."""
        settings = self.settings

        options = {
            "model": settings.deepgram_model,
            "language": settings.deepgram_language,
            "smart_format": settings.deepgram_smart_format,
            "punctuate": settings.deepgram_punctuate,
            "profanity_filter": settings.deepgram_profanity_filter,
            "diarize": settings.deepgram_diarize,
            "filler_words": settings.deepgram_filler_words,
            "interim_results": settings.deepgram_interim_results,
            "utterance_end_ms": str(settings.deepgram_utterance_end_ms),
            "vad_events": settings.deepgram_vad_events,
            "endpointing": str(settings.deepgram_endpointing),
            "encoding": "mulaw",  # Twilio uses mulaw
            "sample_rate": settings.input_sample_rate,
            "channels": settings.channels,
        }

        # Merge custom options
        if custom_options:
            options.update(custom_options)

        return LiveOptions(**options)

    async def _on_open(self, *args, **kwargs) -> None:
        """Handle connection open event."""
        self.logger.debug("Deepgram connection opened")

    async def _on_transcript(self, *args, **kwargs) -> None:
        """Handle transcript event from Deepgram."""
        try:
            result = kwargs.get("result") or (args[1] if len(args) > 1 else None)

            if not result:
                return

            # Extract transcript data
            channel = result.channel
            alternatives = channel.alternatives

            if not alternatives:
                return

            alt = alternatives[0]
            transcript_text = alt.transcript

            # Skip empty transcripts
            if not transcript_text.strip():
                return

            is_final = result.is_final
            speech_final = result.speech_final

            # Build words list
            words = []
            for word_info in alt.words:
                words.append(
                    Word(
                        word=word_info.word,
                        start=word_info.start,
                        end=word_info.end,
                        confidence=word_info.confidence,
                        punctuated_word=getattr(word_info, "punctuated_word", None),
                    )
                )

            # Create transcript result
            transcript = TranscriptResult(
                text=transcript_text,
                is_final=is_final,
                confidence=alt.confidence,
                words=words,
                start_time=result.start,
                end_time=result.start + result.duration,
                speech_final=speech_final,
                channel=0,
                duration=result.duration,
            )

            # Create event
            event = TranscriptEvent(
                type="transcript",
                transcript=transcript,
                timestamp=time.time(),
            )

            # Put in queue and emit
            await self._event_queue.put(event)
            await self._emit_event(event)

            self._last_transcript_time = time.time()

            # Log transcript
            log_method = self.logger.info if is_final else self.logger.debug
            log_method(
                "Transcript received",
                text=transcript_text[:100],
                is_final=is_final,
                speech_final=speech_final,
                confidence=round(alt.confidence, 3),
            )

        except Exception as e:
            self.logger.error("Error processing transcript", error=str(e))

    async def _on_speech_started(self, *args, **kwargs) -> None:
        """Handle speech started event (VAD)."""
        self.logger.debug("Speech started")

        event = TranscriptEvent(
            type="vad",
            vad_event=VADEvent.SPEECH_START,
            timestamp=time.time(),
        )

        await self._event_queue.put(event)
        await self._emit_event(event)

    async def _on_utterance_end(self, *args, **kwargs) -> None:
        """Handle utterance end event."""
        self.logger.debug("Utterance ended")

        event = TranscriptEvent(
            type="vad",
            vad_event=VADEvent.SPEECH_END,
            timestamp=time.time(),
        )

        await self._event_queue.put(event)
        await self._emit_event(event)

    async def _on_metadata(self, *args, **kwargs) -> None:
        """Handle metadata event."""
        result = kwargs.get("result") or (args[1] if len(args) > 1 else None)
        if result:
            self.logger.debug("Metadata received", metadata=str(result)[:200])

    async def _on_error(self, *args, **kwargs) -> None:
        """Handle error event."""
        error = kwargs.get("error") or (args[1] if len(args) > 1 else "Unknown error")
        self.logger.error("Deepgram error", error=str(error))

        event = TranscriptEvent(
            type="error",
            error=str(error),
            timestamp=time.time(),
        )

        await self._event_queue.put(event)
        await self._emit_event(event)

    async def _on_close(self, *args, **kwargs) -> None:
        """Handle connection close event."""
        self.logger.info("Deepgram connection closed")
        self._is_connected = False

    async def disconnect(self) -> None:
        """Disconnect from Deepgram."""
        if self._connection:
            try:
                await self._connection.finish()
            except Exception as e:
                self.logger.error("Error disconnecting from Deepgram", error=str(e))
            finally:
                self._connection = None
                self._is_connected = False

        self.logger.info("Disconnected from Deepgram")

    async def send_audio(self, audio: bytes) -> None:
        """Send audio data to Deepgram."""
        if not self._is_connected or not self._connection:
            raise ConnectionError("Not connected to Deepgram")

        try:
            await self._connection.send(audio)
        except Exception as e:
            self.logger.error("Error sending audio to Deepgram", error=str(e))
            raise

    async def stream_transcripts(self) -> AsyncIterator[TranscriptEvent]:
        """Stream transcript events."""
        while self._is_connected:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=5.0,
                )
                yield event

            except asyncio.TimeoutError:
                # No events, continue waiting
                continue

            except Exception as e:
                self.logger.error("Error streaming transcripts", error=str(e))
                break

    async def finalize(self) -> Optional[TranscriptResult]:
        """Signal end of audio and get final transcript."""
        if self._connection:
            try:
                # Send finalize signal
                await self._connection.finish()

                # Wait briefly for final results
                await asyncio.sleep(0.5)

                # Get any remaining events
                final_transcript = None
                while not self._event_queue.empty():
                    event = await self._event_queue.get()
                    if event.transcript and event.transcript.is_final:
                        final_transcript = event.transcript

                return final_transcript

            except Exception as e:
                self.logger.error("Error finalizing Deepgram connection", error=str(e))

        return None


class DeepgramStreamManager:
    """
    Manages multiple Deepgram streaming sessions.

    Provides connection pooling and session management.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or get_settings().deepgram_api_key
        self._sessions: dict[str, DeepgramAdapter] = {}
        self.logger = logger.bind(component="deepgram_manager")

    async def create_session(
        self,
        session_id: str,
        options: Optional[dict] = None,
    ) -> DeepgramAdapter:
        """Create a new ASR session."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        adapter = DeepgramAdapter(self.api_key)
        await adapter.connect(session_id, options)

        self._sessions[session_id] = adapter
        self.logger.info("Created ASR session", session_id=session_id)

        return adapter

    async def get_session(self, session_id: str) -> Optional[DeepgramAdapter]:
        """Get an existing session."""
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str) -> None:
        """Close and remove a session."""
        adapter = self._sessions.pop(session_id, None)
        if adapter:
            await adapter.disconnect()
            self.logger.info("Closed ASR session", session_id=session_id)

    async def close_all(self) -> None:
        """Close all sessions."""
        for session_id in list(self._sessions.keys()):
            await self.close_session(session_id)

    @property
    def active_sessions(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

"""
Transcription Service Module

This module provides transcription services for call recordings, supporting
multiple providers, diarization, and post-processing capabilities.
"""

import asyncio
import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from .base import (
    Recording,
    Transcription,
    TranscriptSegment,
    TranscriptionStatus,
    TranscriptionProvider,
    TranscriptionError,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Transcription Provider Interface
# =============================================================================


class TranscriptionService(ABC):
    """Abstract base class for transcription providers."""

    def __init__(
        self,
        api_key: str,
        **kwargs,
    ):
        """
        Initialize transcription service.

        Args:
            api_key: API key for the service
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs

    @property
    @abstractmethod
    def provider(self) -> TranscriptionProvider:
        """Get provider type."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_url: str,
        language: str = "en-US",
        diarization: bool = True,
        punctuation: bool = True,
        profanity_filter: bool = False,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Start transcription job.

        Args:
            audio_url: URL to audio file
            language: Language code
            diarization: Enable speaker diarization
            punctuation: Enable automatic punctuation
            profanity_filter: Filter profanity

        Returns:
            Tuple of (job_id, status)
        """
        pass

    @abstractmethod
    async def get_status(self, job_id: str) -> str:
        """
        Get transcription job status.

        Args:
            job_id: Job ID

        Returns:
            Status string
        """
        pass

    @abstractmethod
    async def get_result(self, job_id: str) -> Transcription:
        """
        Get transcription result.

        Args:
            job_id: Job ID

        Returns:
            Transcription object
        """
        pass

    @abstractmethod
    async def transcribe_sync(
        self,
        audio_data: bytes,
        language: str = "en-US",
        **kwargs,
    ) -> Transcription:
        """
        Transcribe audio synchronously.

        Args:
            audio_data: Audio data as bytes
            language: Language code

        Returns:
            Transcription object
        """
        pass


# =============================================================================
# Deepgram Provider
# =============================================================================


class DeepgramTranscriptionService(TranscriptionService):
    """Deepgram transcription service."""

    def __init__(
        self,
        api_key: str,
        **kwargs,
    ):
        """Initialize Deepgram service."""
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api.deepgram.com/v1"
        self.model = kwargs.get("model", "nova-2")

    @property
    def provider(self) -> TranscriptionProvider:
        return TranscriptionProvider.DEEPGRAM

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to Deepgram API."""
        import aiohttp

        url = f"{self.base_url}/{endpoint}"
        request_headers = {
            "Authorization": f"Token {self.api_key}",
        }
        if headers:
            request_headers.update(headers)

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                data=data,
                headers=request_headers,
            ) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise TranscriptionError(f"Deepgram API error: {response.status} - {text}")
                return await response.json()

    async def transcribe(
        self,
        audio_url: str,
        language: str = "en-US",
        diarization: bool = True,
        punctuation: bool = True,
        profanity_filter: bool = False,
        **kwargs,
    ) -> Tuple[str, str]:
        """Start async transcription with Deepgram."""
        params = {
            "model": self.model,
            "language": language,
            "punctuate": str(punctuation).lower(),
            "diarize": str(diarization).lower(),
            "profanity_filter": str(profanity_filter).lower(),
            "utterances": "true",
            "smart_format": "true",
        }

        query_string = "&".join(f"{k}={v}" for k, v in params.items())

        result = await self._make_request(
            "POST",
            f"listen?{query_string}",
            data=json.dumps({"url": audio_url}),
            headers={"Content-Type": "application/json"},
        )

        # Deepgram returns results immediately for pre-recorded
        request_id = result.get("metadata", {}).get("request_id", str(uuid.uuid4()))
        return request_id, "completed"

    async def get_status(self, job_id: str) -> str:
        """Get job status (Deepgram is synchronous for pre-recorded)."""
        return "completed"

    async def get_result(self, job_id: str) -> Transcription:
        """Get result (stored from transcribe call)."""
        # For async Deepgram, results are stored and retrieved by job_id
        raise NotImplementedError("Use transcribe_sync for Deepgram pre-recorded audio")

    async def transcribe_sync(
        self,
        audio_data: bytes,
        language: str = "en-US",
        diarization: bool = True,
        punctuation: bool = True,
        **kwargs,
    ) -> Transcription:
        """Transcribe audio synchronously with Deepgram."""
        params = {
            "model": self.model,
            "language": language,
            "punctuate": str(punctuation).lower(),
            "diarize": str(diarization).lower(),
            "utterances": "true",
            "smart_format": "true",
        }

        query_string = "&".join(f"{k}={v}" for k, v in params.items())

        result = await self._make_request(
            "POST",
            f"listen?{query_string}",
            data=audio_data,
            headers={"Content-Type": "audio/wav"},
        )

        return self._parse_response(result, kwargs.get("recording_id", ""), kwargs.get("organization_id", ""))

    def _parse_response(
        self,
        response: Dict[str, Any],
        recording_id: str,
        organization_id: str,
    ) -> Transcription:
        """Parse Deepgram response into Transcription object."""
        results = response.get("results", {})
        channels = results.get("channels", [])

        if not channels:
            return Transcription(
                id=f"trx_{uuid.uuid4().hex[:24]}",
                recording_id=recording_id,
                organization_id=organization_id,
                status=TranscriptionStatus.COMPLETED,
                provider=TranscriptionProvider.DEEPGRAM,
                full_text="",
                segments=[],
            )

        # Get first channel alternatives
        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return Transcription(
                id=f"trx_{uuid.uuid4().hex[:24]}",
                recording_id=recording_id,
                organization_id=organization_id,
                status=TranscriptionStatus.COMPLETED,
                provider=TranscriptionProvider.DEEPGRAM,
            )

        best_alternative = alternatives[0]
        full_text = best_alternative.get("transcript", "")
        words = best_alternative.get("words", [])

        # Build segments from utterances or words
        segments = []
        utterances = results.get("utterances", [])

        if utterances:
            for utt in utterances:
                speaker = utt.get("speaker", 0)
                speaker_label = "agent" if speaker == 0 else "caller"

                segment = TranscriptSegment(
                    id=f"seg_{uuid.uuid4().hex[:12]}",
                    text=utt.get("transcript", ""),
                    start_time=utt.get("start", 0.0),
                    end_time=utt.get("end", 0.0),
                    speaker=speaker_label,
                    speaker_id=str(speaker),
                    confidence=utt.get("confidence", 1.0),
                    words=[
                        {
                            "word": w.get("word", ""),
                            "start": w.get("start", 0.0),
                            "end": w.get("end", 0.0),
                            "confidence": w.get("confidence", 1.0),
                        }
                        for w in utt.get("words", [])
                    ],
                )
                segments.append(segment)
        else:
            # Create segments from word groupings
            current_segment_words = []
            segment_start = 0.0

            for i, word in enumerate(words):
                current_segment_words.append(word)

                # Check if this is a sentence end or last word
                is_sentence_end = word.get("punctuated_word", "").endswith((".", "?", "!"))
                is_last = i == len(words) - 1

                if is_sentence_end or is_last:
                    if current_segment_words:
                        segment = TranscriptSegment(
                            id=f"seg_{uuid.uuid4().hex[:12]}",
                            text=" ".join(w.get("punctuated_word", w.get("word", "")) for w in current_segment_words),
                            start_time=segment_start,
                            end_time=word.get("end", 0.0),
                            speaker=str(word.get("speaker", "unknown")),
                            confidence=sum(w.get("confidence", 1.0) for w in current_segment_words) / len(current_segment_words),
                            words=[
                                {
                                    "word": w.get("punctuated_word", w.get("word", "")),
                                    "start": w.get("start", 0.0),
                                    "end": w.get("end", 0.0),
                                    "confidence": w.get("confidence", 1.0),
                                }
                                for w in current_segment_words
                            ],
                        )
                        segments.append(segment)

                        # Reset for next segment
                        current_segment_words = []
                        if i + 1 < len(words):
                            segment_start = words[i + 1].get("start", 0.0)

        # Calculate statistics
        word_count = len(words)
        speakers = set(s.speaker for s in segments)
        avg_confidence = sum(s.confidence for s in segments) / len(segments) if segments else 0.0

        detected_language = response.get("results", {}).get("channels", [{}])[0].get("detected_language")

        return Transcription(
            id=f"trx_{uuid.uuid4().hex[:24]}",
            recording_id=recording_id,
            organization_id=organization_id,
            status=TranscriptionStatus.COMPLETED,
            provider=TranscriptionProvider.DEEPGRAM,
            provider_job_id=response.get("metadata", {}).get("request_id"),
            detected_language=detected_language,
            full_text=full_text,
            segments=segments,
            word_count=word_count,
            speaker_count=len(speakers),
            average_confidence=avg_confidence,
            completed_at=datetime.utcnow(),
        )


# =============================================================================
# AssemblyAI Provider
# =============================================================================


class AssemblyAITranscriptionService(TranscriptionService):
    """AssemblyAI transcription service."""

    def __init__(
        self,
        api_key: str,
        **kwargs,
    ):
        """Initialize AssemblyAI service."""
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api.assemblyai.com/v2"

    @property
    def provider(self) -> TranscriptionProvider:
        return TranscriptionProvider.ASSEMBLY_AI

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to AssemblyAI API."""
        import aiohttp

        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                json=data,
                headers=headers,
            ) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise TranscriptionError(f"AssemblyAI API error: {response.status} - {text}")
                return await response.json()

    async def transcribe(
        self,
        audio_url: str,
        language: str = "en-US",
        diarization: bool = True,
        punctuation: bool = True,
        profanity_filter: bool = False,
        **kwargs,
    ) -> Tuple[str, str]:
        """Start async transcription with AssemblyAI."""
        data = {
            "audio_url": audio_url,
            "language_code": language.split("-")[0],  # AssemblyAI uses 2-letter codes
            "speaker_labels": diarization,
            "punctuate": punctuation,
            "filter_profanity": profanity_filter,
            "auto_chapters": kwargs.get("auto_chapters", False),
            "entity_detection": kwargs.get("entity_detection", False),
            "sentiment_analysis": kwargs.get("sentiment_analysis", False),
        }

        result = await self._make_request("POST", "transcript", data)
        return result.get("id", ""), result.get("status", "queued")

    async def get_status(self, job_id: str) -> str:
        """Get transcription job status."""
        result = await self._make_request("GET", f"transcript/{job_id}")
        return result.get("status", "unknown")

    async def get_result(self, job_id: str) -> Transcription:
        """Get transcription result."""
        result = await self._make_request("GET", f"transcript/{job_id}")

        if result.get("status") != "completed":
            raise TranscriptionError(f"Transcription not complete: {result.get('status')}")

        return self._parse_response(result)

    async def transcribe_sync(
        self,
        audio_data: bytes,
        language: str = "en-US",
        **kwargs,
    ) -> Transcription:
        """Transcribe audio synchronously (upload then poll)."""
        import aiohttp

        # Upload audio first
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/upload",
                data=audio_data,
                headers={"Authorization": self.api_key},
            ) as response:
                if response.status >= 400:
                    raise TranscriptionError("Failed to upload audio to AssemblyAI")
                upload_result = await response.json()
                audio_url = upload_result.get("upload_url")

        # Start transcription
        job_id, status = await self.transcribe(
            audio_url=audio_url,
            language=language,
            **kwargs,
        )

        # Poll for completion
        max_wait = 300  # 5 minutes
        poll_interval = 3
        elapsed = 0

        while elapsed < max_wait:
            status = await self.get_status(job_id)
            if status == "completed":
                return await self.get_result(job_id)
            elif status == "error":
                raise TranscriptionError("Transcription failed")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TranscriptionError("Transcription timeout")

    def _parse_response(self, response: Dict[str, Any]) -> Transcription:
        """Parse AssemblyAI response into Transcription object."""
        segments = []
        words = response.get("words", [])

        # Parse utterances if available
        utterances = response.get("utterances", [])
        if utterances:
            for utt in utterances:
                speaker_label = "agent" if utt.get("speaker") == "A" else "caller"
                segment = TranscriptSegment(
                    id=f"seg_{uuid.uuid4().hex[:12]}",
                    text=utt.get("text", ""),
                    start_time=utt.get("start", 0) / 1000.0,
                    end_time=utt.get("end", 0) / 1000.0,
                    speaker=speaker_label,
                    speaker_id=utt.get("speaker", "unknown"),
                    confidence=utt.get("confidence", 1.0),
                    words=[
                        {
                            "word": w.get("text", ""),
                            "start": w.get("start", 0) / 1000.0,
                            "end": w.get("end", 0) / 1000.0,
                            "confidence": w.get("confidence", 1.0),
                        }
                        for w in utt.get("words", [])
                    ],
                )

                # Add sentiment if available
                if "sentiment" in utt:
                    segment.sentiment = utt["sentiment"]
                    segment.sentiment_score = utt.get("sentiment_score")

                segments.append(segment)

        # Calculate statistics
        speakers = set(s.speaker for s in segments)
        avg_confidence = response.get("confidence", 0.0)

        # Get summary and topics if available
        summary = None
        chapters = response.get("chapters", [])
        if chapters:
            summary = " ".join(c.get("summary", "") for c in chapters)

        key_topics = []
        entities = response.get("entities", [])
        for entity in entities:
            if entity.get("entity_type") in ["topic", "key_phrase"]:
                key_topics.append(entity.get("text", ""))

        # Get action items from auto-chapters
        action_items = []
        for chapter in chapters:
            if "action" in chapter.get("headline", "").lower():
                action_items.append(chapter.get("summary", ""))

        return Transcription(
            id=f"trx_{uuid.uuid4().hex[:24]}",
            recording_id="",  # Will be set by caller
            organization_id="",  # Will be set by caller
            status=TranscriptionStatus.COMPLETED,
            provider=TranscriptionProvider.ASSEMBLY_AI,
            provider_job_id=response.get("id"),
            language=response.get("language_code", "en"),
            full_text=response.get("text", ""),
            segments=segments,
            word_count=len(words),
            speaker_count=len(speakers),
            average_confidence=avg_confidence,
            summary=summary,
            key_topics=key_topics[:10],
            action_items=action_items,
            completed_at=datetime.utcnow(),
        )


# =============================================================================
# OpenAI Whisper Provider
# =============================================================================


class WhisperTranscriptionService(TranscriptionService):
    """OpenAI Whisper transcription service."""

    def __init__(
        self,
        api_key: str,
        **kwargs,
    ):
        """Initialize Whisper service."""
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api.openai.com/v1"
        self.model = kwargs.get("model", "whisper-1")

    @property
    def provider(self) -> TranscriptionProvider:
        return TranscriptionProvider.OPENAI_WHISPER

    async def transcribe(
        self,
        audio_url: str,
        language: str = "en-US",
        diarization: bool = True,
        punctuation: bool = True,
        profanity_filter: bool = False,
        **kwargs,
    ) -> Tuple[str, str]:
        """Whisper doesn't support async - use transcribe_sync."""
        raise NotImplementedError("Use transcribe_sync for Whisper")

    async def get_status(self, job_id: str) -> str:
        """Whisper is synchronous."""
        return "completed"

    async def get_result(self, job_id: str) -> Transcription:
        """Whisper is synchronous."""
        raise NotImplementedError("Use transcribe_sync for Whisper")

    async def transcribe_sync(
        self,
        audio_data: bytes,
        language: str = "en-US",
        **kwargs,
    ) -> Transcription:
        """Transcribe audio with Whisper."""
        import aiohttp

        form = aiohttp.FormData()
        form.add_field(
            "file",
            audio_data,
            filename="audio.wav",
            content_type="audio/wav",
        )
        form.add_field("model", self.model)
        form.add_field("response_format", "verbose_json")

        if language:
            form.add_field("language", language.split("-")[0])

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/audio/transcriptions",
                data=form,
                headers={"Authorization": f"Bearer {self.api_key}"},
            ) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise TranscriptionError(f"Whisper API error: {response.status} - {text}")
                result = await response.json()

        return self._parse_response(
            result,
            kwargs.get("recording_id", ""),
            kwargs.get("organization_id", ""),
        )

    def _parse_response(
        self,
        response: Dict[str, Any],
        recording_id: str,
        organization_id: str,
    ) -> Transcription:
        """Parse Whisper response into Transcription object."""
        full_text = response.get("text", "")
        segments_data = response.get("segments", [])

        segments = []
        for seg in segments_data:
            segment = TranscriptSegment(
                id=f"seg_{uuid.uuid4().hex[:12]}",
                text=seg.get("text", "").strip(),
                start_time=seg.get("start", 0.0),
                end_time=seg.get("end", 0.0),
                speaker="unknown",  # Whisper doesn't provide diarization
                confidence=seg.get("no_speech_prob", 0.0),
                words=[
                    {
                        "word": w.get("word", ""),
                        "start": w.get("start", 0.0),
                        "end": w.get("end", 0.0),
                        "probability": w.get("probability", 1.0),
                    }
                    for w in seg.get("words", [])
                ],
            )
            segments.append(segment)

        # Count words
        word_count = sum(len(seg.get("words", [])) for seg in segments_data)
        if word_count == 0:
            word_count = len(full_text.split())

        return Transcription(
            id=f"trx_{uuid.uuid4().hex[:24]}",
            recording_id=recording_id,
            organization_id=organization_id,
            status=TranscriptionStatus.COMPLETED,
            provider=TranscriptionProvider.OPENAI_WHISPER,
            language=response.get("language", "en"),
            detected_language=response.get("language"),
            full_text=full_text,
            segments=segments,
            word_count=word_count,
            speaker_count=1,  # Whisper doesn't diarize
            average_confidence=1.0 - (sum(s.get("no_speech_prob", 0) for s in segments_data) / max(len(segments_data), 1)),
            completed_at=datetime.utcnow(),
        )


# =============================================================================
# Transcription Manager
# =============================================================================


class TranscriptionManager:
    """
    Manages transcription services and operations.
    """

    def __init__(self):
        """Initialize transcription manager."""
        self._services: Dict[TranscriptionProvider, TranscriptionService] = {}
        self._default_provider: Optional[TranscriptionProvider] = None

        # Transcription storage (in production, use database)
        self._transcriptions: Dict[str, Transcription] = {}

    def register_service(
        self,
        service: TranscriptionService,
        is_default: bool = False,
    ) -> None:
        """
        Register a transcription service.

        Args:
            service: Transcription service instance
            is_default: Set as default service
        """
        self._services[service.provider] = service
        if is_default or self._default_provider is None:
            self._default_provider = service.provider

    def get_service(
        self,
        provider: Optional[TranscriptionProvider] = None,
    ) -> TranscriptionService:
        """
        Get transcription service.

        Args:
            provider: Provider type (uses default if None)

        Returns:
            Transcription service
        """
        if provider:
            if provider not in self._services:
                raise TranscriptionError(f"Provider not registered: {provider}")
            return self._services[provider]

        if not self._default_provider:
            raise TranscriptionError("No transcription service registered")

        return self._services[self._default_provider]

    async def transcribe_recording(
        self,
        recording: Recording,
        audio_data: Optional[bytes] = None,
        audio_url: Optional[str] = None,
        provider: Optional[TranscriptionProvider] = None,
        language: str = "en-US",
        diarization: bool = True,
        **kwargs,
    ) -> Transcription:
        """
        Transcribe a recording.

        Args:
            recording: Recording object
            audio_data: Audio data as bytes
            audio_url: URL to audio file
            provider: Transcription provider
            language: Language code
            diarization: Enable speaker diarization
            **kwargs: Additional provider options

        Returns:
            Transcription object
        """
        service = self.get_service(provider)

        try:
            if audio_data:
                transcription = await service.transcribe_sync(
                    audio_data=audio_data,
                    language=language,
                    diarization=diarization,
                    recording_id=recording.id,
                    organization_id=recording.organization_id,
                    **kwargs,
                )
            elif audio_url:
                job_id, status = await service.transcribe(
                    audio_url=audio_url,
                    language=language,
                    diarization=diarization,
                    **kwargs,
                )
                transcription = Transcription(
                    id=f"trx_{uuid.uuid4().hex[:24]}",
                    recording_id=recording.id,
                    organization_id=recording.organization_id,
                    status=TranscriptionStatus.IN_PROGRESS,
                    provider=service.provider,
                    provider_job_id=job_id,
                    language=language,
                )
            else:
                raise TranscriptionError("Either audio_data or audio_url required")

            # Update recording reference
            recording.transcription_id = transcription.id

            # Store transcription
            self._transcriptions[transcription.id] = transcription

            logger.info(f"Created transcription {transcription.id} for recording {recording.id}")
            return transcription

        except Exception as e:
            logger.error(f"Transcription failed for recording {recording.id}: {e}")
            raise TranscriptionError(f"Transcription failed: {e}")

    async def get_transcription(
        self,
        transcription_id: str,
    ) -> Optional[Transcription]:
        """
        Get a transcription by ID.

        Args:
            transcription_id: Transcription ID

        Returns:
            Transcription or None
        """
        return self._transcriptions.get(transcription_id)

    async def poll_transcription(
        self,
        transcription: Transcription,
    ) -> Transcription:
        """
        Poll for transcription completion.

        Args:
            transcription: Transcription object

        Returns:
            Updated transcription
        """
        if transcription.status == TranscriptionStatus.COMPLETED:
            return transcription

        service = self.get_service(transcription.provider)
        status = await service.get_status(transcription.provider_job_id)

        if status == "completed":
            result = await service.get_result(transcription.provider_job_id)
            result.id = transcription.id
            result.recording_id = transcription.recording_id
            result.organization_id = transcription.organization_id
            self._transcriptions[transcription.id] = result
            return result
        elif status == "error" or status == "failed":
            transcription.status = TranscriptionStatus.FAILED
            transcription.error_message = "Transcription failed"
            return transcription

        return transcription

    async def wait_for_completion(
        self,
        transcription: Transcription,
        timeout_seconds: int = 300,
        poll_interval: int = 3,
    ) -> Transcription:
        """
        Wait for transcription to complete.

        Args:
            transcription: Transcription object
            timeout_seconds: Maximum wait time
            poll_interval: Seconds between polls

        Returns:
            Completed transcription
        """
        if transcription.status == TranscriptionStatus.COMPLETED:
            return transcription

        elapsed = 0
        while elapsed < timeout_seconds:
            transcription = await self.poll_transcription(transcription)

            if transcription.status == TranscriptionStatus.COMPLETED:
                return transcription
            elif transcription.status == TranscriptionStatus.FAILED:
                raise TranscriptionError(transcription.error_message or "Transcription failed")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TranscriptionError("Transcription timeout")

    def search_transcriptions(
        self,
        query: str,
        organization_id: Optional[str] = None,
    ) -> List[Tuple[Transcription, List[TranscriptSegment]]]:
        """
        Search transcriptions for text.

        Args:
            query: Search query
            organization_id: Filter by organization

        Returns:
            List of (transcription, matching_segments) tuples
        """
        results = []
        query_lower = query.lower()

        for transcription in self._transcriptions.values():
            if organization_id and transcription.organization_id != organization_id:
                continue

            matching_segments = []
            for segment in transcription.segments:
                if query_lower in segment.text.lower():
                    matching_segments.append(segment)

            if matching_segments:
                results.append((transcription, matching_segments))

        return results


# =============================================================================
# Transcription Post-Processing
# =============================================================================


class TranscriptionPostProcessor:
    """
    Post-processing utilities for transcriptions.
    """

    @staticmethod
    def extract_key_phrases(
        transcription: Transcription,
        max_phrases: int = 10,
    ) -> List[str]:
        """
        Extract key phrases from transcription.

        Args:
            transcription: Transcription object
            max_phrases: Maximum phrases to extract

        Returns:
            List of key phrases
        """
        # Simple implementation - could be enhanced with NLP
        text = transcription.full_text.lower()
        words = text.split()

        # Count word frequency
        word_freq: Dict[str, int] = {}
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                      "have", "has", "had", "do", "does", "did", "will", "would", "could",
                      "should", "may", "might", "must", "shall", "can", "need", "dare",
                      "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
                      "from", "up", "about", "into", "over", "after", "beneath", "under",
                      "above", "i", "you", "he", "she", "it", "we", "they", "me", "him",
                      "her", "us", "them", "my", "your", "his", "its", "our", "their",
                      "this", "that", "these", "those", "and", "but", "or", "not", "so",
                      "if", "then", "because", "as", "until", "while", "yeah", "okay",
                      "um", "uh", "oh", "like", "just", "right", "well"}

        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and len(clean_word) > 2 and clean_word not in stop_words:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

        # Get top phrases
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_phrases]]

    @staticmethod
    def calculate_talk_ratios(
        transcription: Transcription,
    ) -> Dict[str, float]:
        """
        Calculate talk time ratios for each speaker.

        Args:
            transcription: Transcription object

        Returns:
            Dictionary of speaker -> talk ratio
        """
        speaker_times: Dict[str, float] = {}

        for segment in transcription.segments:
            duration = segment.end_time - segment.start_time
            speaker_times[segment.speaker] = speaker_times.get(segment.speaker, 0.0) + duration

        total_time = sum(speaker_times.values())
        if total_time == 0:
            return {}

        return {speaker: time / total_time for speaker, time in speaker_times.items()}

    @staticmethod
    def format_as_dialogue(
        transcription: Transcription,
        include_timestamps: bool = False,
    ) -> str:
        """
        Format transcription as readable dialogue.

        Args:
            transcription: Transcription object
            include_timestamps: Include timestamps in output

        Returns:
            Formatted dialogue string
        """
        lines = []
        current_speaker = None

        for segment in transcription.segments:
            if segment.speaker != current_speaker:
                current_speaker = segment.speaker
                speaker_label = current_speaker.upper()

                if include_timestamps:
                    timestamp = f"[{segment.start_time:.1f}s]"
                    lines.append(f"\n{speaker_label} {timestamp}:")
                else:
                    lines.append(f"\n{speaker_label}:")

            lines.append(f"  {segment.text}")

        return "\n".join(lines).strip()


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "TranscriptionService",
    "DeepgramTranscriptionService",
    "AssemblyAITranscriptionService",
    "WhisperTranscriptionService",
    "TranscriptionManager",
    "TranscriptionPostProcessor",
]

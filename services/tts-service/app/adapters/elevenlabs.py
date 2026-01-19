"""ElevenLabs TTS adapter with real-time streaming support."""

import asyncio
import struct
from typing import AsyncIterator, Optional

import httpx
import structlog

from app.adapters.base import TTSAdapter, AudioChunk, VoiceInfo
from app.config import get_settings

logger = structlog.get_logger()


class ElevenLabsAdapter(TTSAdapter):
    """
    ElevenLabs TTS adapter using their streaming API.

    Features:
    - Real-time streaming audio synthesis
    - Multiple voices and models
    - Voice cloning support
    - Low latency mode (turbo v2)
    """

    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self, api_key: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.elevenlabs_api_key

        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")

        self.settings = settings
        self.default_voice_id = settings.elevenlabs_voice_id
        self.model_id = settings.elevenlabs_model_id

        self._client: Optional[httpx.AsyncClient] = None
        self.logger = logger.bind(adapter="elevenlabs")

    @property
    def name(self) -> str:
        return "elevenlabs"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """Synthesize text to audio (non-streaming)."""
        voice = voice_id or self.default_voice_id

        client = await self._get_client()

        # Build request
        payload = {
            "text": text,
            "model_id": kwargs.get("model_id", self.model_id),
            "voice_settings": {
                "stability": kwargs.get("stability", self.settings.elevenlabs_stability),
                "similarity_boost": kwargs.get(
                    "similarity_boost", self.settings.elevenlabs_similarity_boost
                ),
                "style": kwargs.get("style", self.settings.elevenlabs_style),
                "use_speaker_boost": kwargs.get(
                    "use_speaker_boost", self.settings.elevenlabs_use_speaker_boost
                ),
            },
        }

        output_format = kwargs.get("output_format", self.settings.output_format)

        try:
            response = await client.post(
                f"/text-to-speech/{voice}",
                json=payload,
                params={"output_format": output_format},
            )
            response.raise_for_status()

            self.logger.info(
                "Synthesized audio",
                text_length=len(text),
                voice_id=voice,
                audio_size=len(response.content),
            )

            return response.content

        except httpx.HTTPError as e:
            self.logger.error("ElevenLabs API error", error=str(e))
            raise

    async def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[AudioChunk]:
        """Stream synthesized audio chunks."""
        voice = voice_id or self.default_voice_id

        client = await self._get_client()

        # Build request
        payload = {
            "text": text,
            "model_id": kwargs.get("model_id", self.model_id),
            "voice_settings": {
                "stability": kwargs.get("stability", self.settings.elevenlabs_stability),
                "similarity_boost": kwargs.get(
                    "similarity_boost", self.settings.elevenlabs_similarity_boost
                ),
                "style": kwargs.get("style", self.settings.elevenlabs_style),
                "use_speaker_boost": kwargs.get(
                    "use_speaker_boost", self.settings.elevenlabs_use_speaker_boost
                ),
            },
        }

        # Use PCM output for easier processing
        # ulaw_8000 for Twilio compatibility
        output_format = kwargs.get("output_format", "ulaw_8000")

        try:
            async with client.stream(
                "POST",
                f"/text-to-speech/{voice}/stream",
                json=payload,
                params={
                    "output_format": output_format,
                    "optimize_streaming_latency": self.settings.latency_optimization,
                },
            ) as response:
                response.raise_for_status()

                sequence = 0
                async for chunk in response.aiter_bytes(chunk_size=self.settings.chunk_size):
                    if chunk:
                        sequence += 1
                        yield AudioChunk(
                            data=chunk,
                            sample_rate=8000,  # ulaw_8000
                            encoding="mulaw",
                            sequence=sequence,
                        )

                # Send final chunk
                yield AudioChunk(
                    data=b"",
                    sample_rate=8000,
                    encoding="mulaw",
                    sequence=sequence + 1,
                    is_final=True,
                )

            self.logger.info(
                "Streamed audio",
                text_length=len(text),
                voice_id=voice,
                chunks=sequence,
            )

        except httpx.HTTPError as e:
            self.logger.error("ElevenLabs streaming error", error=str(e))
            raise

    async def synthesize_stream_text_chunks(
        self,
        text_chunks: AsyncIterator[str],
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[AudioChunk]:
        """
        Stream audio for streaming text input.

        Useful for LLM streaming responses - sends text chunks as they arrive
        and receives audio chunks back.
        """
        voice = voice_id or self.default_voice_id

        # ElevenLabs supports input streaming via WebSocket
        # For now, buffer small chunks and synthesize
        buffer = ""
        sequence = 0

        async for text_chunk in text_chunks:
            buffer += text_chunk

            # Synthesize when we have a sentence or enough text
            if self._should_synthesize(buffer):
                async for audio_chunk in self.synthesize_stream(
                    buffer, voice_id=voice, **kwargs
                ):
                    audio_chunk.sequence = sequence
                    sequence += 1
                    yield audio_chunk
                buffer = ""

        # Synthesize remaining text
        if buffer.strip():
            async for audio_chunk in self.synthesize_stream(
                buffer, voice_id=voice, **kwargs
            ):
                audio_chunk.sequence = sequence
                sequence += 1
                audio_chunk.is_final = True
                yield audio_chunk

    def _should_synthesize(self, text: str) -> bool:
        """Check if we should synthesize the buffered text."""
        # Synthesize on sentence boundaries
        if any(text.rstrip().endswith(p) for p in [".", "!", "?", ":", ";"]):
            return len(text) > 20  # Minimum length

        # Or if buffer is getting large
        return len(text) > 100

    async def get_voices(self) -> list[VoiceInfo]:
        """Get available voices."""
        client = await self._get_client()

        try:
            response = await client.get("/voices")
            response.raise_for_status()

            data = response.json()
            voices = []

            for voice in data.get("voices", []):
                voices.append(
                    VoiceInfo(
                        voice_id=voice["voice_id"],
                        name=voice["name"],
                        description=voice.get("description"),
                        labels=voice.get("labels"),
                        preview_url=voice.get("preview_url"),
                    )
                )

            return voices

        except httpx.HTTPError as e:
            self.logger.error("Failed to get voices", error=str(e))
            raise

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    # Voice Cloning Methods

    async def clone_voice(
        self,
        name: str,
        audio_files: list[bytes],
        description: str = "",
        labels: Optional[dict] = None,
    ) -> VoiceInfo:
        """
        Create a cloned voice from audio samples.

        Args:
            name: Name for the cloned voice
            audio_files: List of audio file bytes (WAV, MP3, etc.)
            description: Optional description of the voice
            labels: Optional labels (e.g., {"accent": "american", "gender": "male"})

        Returns:
            VoiceInfo for the newly created voice

        Notes:
            - Requires at least 1 audio sample
            - Best results with 3+ samples of 30+ seconds each
            - Audio should be clear speech without background noise
        """
        if not audio_files:
            raise ValueError("At least one audio file is required for voice cloning")

        client = await self._get_client()

        # Build multipart form data
        files = []
        for i, audio_data in enumerate(audio_files):
            files.append(("files", (f"sample_{i}.mp3", audio_data, "audio/mpeg")))

        data = {
            "name": name,
            "description": description,
        }

        if labels:
            data["labels"] = str(labels)

        try:
            # Remove content-type header for multipart
            headers = {
                "xi-api-key": self.api_key,
            }

            response = await client.post(
                "/voices/add",
                data=data,
                files=files,
                headers=headers,
            )
            response.raise_for_status()

            result = response.json()

            self.logger.info(
                "Voice cloned successfully",
                voice_id=result.get("voice_id"),
                name=name,
            )

            return VoiceInfo(
                voice_id=result["voice_id"],
                name=name,
                description=description,
                labels=labels,
            )

        except httpx.HTTPError as e:
            self.logger.error("Failed to clone voice", error=str(e))
            raise

    async def clone_voice_instant(
        self,
        name: str,
        audio_files: list[bytes],
        description: str = "",
    ) -> VoiceInfo:
        """
        Create an instant voice clone (faster, less samples needed).

        This uses ElevenLabs' instant voice cloning which requires
        only a short sample but may have slightly lower quality.

        Args:
            name: Name for the cloned voice
            audio_files: List of audio file bytes (can be just one)
            description: Optional description

        Returns:
            VoiceInfo for the newly created voice
        """
        return await self.clone_voice(
            name=name,
            audio_files=audio_files,
            description=description,
        )

    async def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a cloned voice.

        Args:
            voice_id: ID of the voice to delete

        Returns:
            True if deletion was successful
        """
        client = await self._get_client()

        try:
            response = await client.delete(f"/voices/{voice_id}")
            response.raise_for_status()

            self.logger.info("Voice deleted", voice_id=voice_id)
            return True

        except httpx.HTTPError as e:
            self.logger.error("Failed to delete voice", voice_id=voice_id, error=str(e))
            raise

    async def edit_voice(
        self,
        voice_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[dict] = None,
    ) -> VoiceInfo:
        """
        Edit a cloned voice's metadata.

        Args:
            voice_id: ID of the voice to edit
            name: New name (optional)
            description: New description (optional)
            labels: New labels (optional)

        Returns:
            Updated VoiceInfo
        """
        client = await self._get_client()

        data = {}
        if name:
            data["name"] = name
        if description:
            data["description"] = description
        if labels:
            data["labels"] = str(labels)

        try:
            response = await client.post(
                f"/voices/{voice_id}/edit",
                data=data,
            )
            response.raise_for_status()

            # Get updated voice info
            return await self.get_voice(voice_id)

        except httpx.HTTPError as e:
            self.logger.error("Failed to edit voice", voice_id=voice_id, error=str(e))
            raise

    async def get_voice(self, voice_id: str) -> VoiceInfo:
        """
        Get information about a specific voice.

        Args:
            voice_id: ID of the voice

        Returns:
            VoiceInfo for the voice
        """
        client = await self._get_client()

        try:
            response = await client.get(f"/voices/{voice_id}")
            response.raise_for_status()

            voice = response.json()

            return VoiceInfo(
                voice_id=voice["voice_id"],
                name=voice["name"],
                description=voice.get("description"),
                labels=voice.get("labels"),
                preview_url=voice.get("preview_url"),
            )

        except httpx.HTTPError as e:
            self.logger.error("Failed to get voice", voice_id=voice_id, error=str(e))
            raise

    async def add_voice_samples(
        self,
        voice_id: str,
        audio_files: list[bytes],
    ) -> VoiceInfo:
        """
        Add additional audio samples to an existing cloned voice.

        Args:
            voice_id: ID of the voice to add samples to
            audio_files: List of audio file bytes

        Returns:
            Updated VoiceInfo
        """
        if not audio_files:
            raise ValueError("At least one audio file is required")

        client = await self._get_client()

        files = []
        for i, audio_data in enumerate(audio_files):
            files.append(("files", (f"sample_{i}.mp3", audio_data, "audio/mpeg")))

        try:
            headers = {
                "xi-api-key": self.api_key,
            }

            response = await client.post(
                f"/voices/{voice_id}/edit",
                files=files,
                headers=headers,
            )
            response.raise_for_status()

            self.logger.info("Voice samples added", voice_id=voice_id, count=len(audio_files))

            return await self.get_voice(voice_id)

        except httpx.HTTPError as e:
            self.logger.error("Failed to add voice samples", voice_id=voice_id, error=str(e))
            raise

    async def get_voice_settings(self, voice_id: str) -> dict:
        """
        Get the voice settings for a specific voice.

        Args:
            voice_id: ID of the voice

        Returns:
            Voice settings dict
        """
        client = await self._get_client()

        try:
            response = await client.get(f"/voices/{voice_id}/settings")
            response.raise_for_status()

            return response.json()

        except httpx.HTTPError as e:
            self.logger.error("Failed to get voice settings", voice_id=voice_id, error=str(e))
            raise

    async def update_voice_settings(
        self,
        voice_id: str,
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None,
        style: Optional[float] = None,
        use_speaker_boost: Optional[bool] = None,
    ) -> dict:
        """
        Update the default settings for a voice.

        Args:
            voice_id: ID of the voice
            stability: Voice stability (0.0 - 1.0)
            similarity_boost: Similarity enhancement (0.0 - 1.0)
            style: Style exaggeration (0.0 - 1.0)
            use_speaker_boost: Enable speaker boost

        Returns:
            Updated settings dict
        """
        client = await self._get_client()

        data = {}
        if stability is not None:
            data["stability"] = stability
        if similarity_boost is not None:
            data["similarity_boost"] = similarity_boost
        if style is not None:
            data["style"] = style
        if use_speaker_boost is not None:
            data["use_speaker_boost"] = use_speaker_boost

        try:
            response = await client.post(
                f"/voices/{voice_id}/settings/edit",
                json=data,
            )
            response.raise_for_status()

            return response.json()

        except httpx.HTTPError as e:
            self.logger.error("Failed to update voice settings", voice_id=voice_id, error=str(e))
            raise


def pcm_to_mulaw(pcm_data: bytes) -> bytes:
    """Convert PCM16 audio to μ-law encoding."""
    # μ-law encoding table
    MULAW_MAX = 0x1FFF
    MULAW_BIAS = 33

    output = bytearray()

    for i in range(0, len(pcm_data), 2):
        if i + 1 >= len(pcm_data):
            break

        # Read 16-bit sample (little-endian)
        sample = struct.unpack("<h", pcm_data[i : i + 2])[0]

        # Get sign
        sign = 0x80 if sample < 0 else 0
        if sample < 0:
            sample = -sample

        # Clip
        if sample > MULAW_MAX:
            sample = MULAW_MAX

        # Add bias
        sample += MULAW_BIAS

        # Find segment
        exponent = 7
        for exp in range(7, -1, -1):
            if sample & (1 << (exp + 7)):
                exponent = exp
                break

        # Build μ-law byte
        mantissa = (sample >> (exponent + 3)) & 0x0F
        mulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF

        output.append(mulaw_byte)

    return bytes(output)

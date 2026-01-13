"""
Builder Engine Python SDK - Voices Resource

This module provides methods for managing voice configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List, BinaryIO

from builderengine.resources.base import BaseResource, PaginatedResponse
from builderengine.models import Voice, VoiceProvider, VoiceConfig
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class VoicesResource(BaseResource):
    """
    Resource for managing voice configurations.

    Voices define how the AI agent sounds during calls. This resource
    provides access to pre-built voices, custom voice cloning, and
    voice configuration management.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> # List available voices
        >>> voices = client.voices.list_library()
        >>> # Preview a voice
        >>> audio = client.voices.preview("voice_abc123", "Hello, world!")
    """

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        provider: Optional[VoiceProvider] = None,
        language: Optional[str] = None,
        gender: Optional[str] = None,
        is_custom: Optional[bool] = None,
    ) -> PaginatedResponse[Voice]:
        """
        List organization's voice configurations.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            provider: Filter by voice provider
            language: Filter by language code
            gender: Filter by gender (male, female, neutral)
            is_custom: Filter for custom/cloned voices

        Returns:
            PaginatedResponse containing Voice objects
        """
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            provider=provider.value if provider else None,
            language=language,
            gender=gender,
            is_custom=is_custom,
        )
        response = self._get(Endpoints.VOICES, params=params)
        return self._parse_paginated_response(response, Voice)

    def get(self, voice_id: str) -> Voice:
        """
        Get a voice by ID.

        Args:
            voice_id: The voice's unique identifier

        Returns:
            Voice object
        """
        path = Endpoints.VOICE.format(voice_id=voice_id)
        response = self._get(path)
        return Voice.from_dict(response)

    def list_library(
        self,
        provider: Optional[VoiceProvider] = None,
        language: Optional[str] = None,
        gender: Optional[str] = None,
        accent: Optional[str] = None,
        use_case: Optional[str] = None,
    ) -> List[Voice]:
        """
        List all voices from the voice library.

        This includes pre-built voices from all providers.

        Args:
            provider: Filter by provider
            language: Filter by language code (e.g., "en-US")
            gender: Filter by gender
            accent: Filter by accent (e.g., "american", "british")
            use_case: Filter by recommended use case

        Returns:
            List of Voice objects

        Example:
            >>> voices = client.voices.list_library(
            ...     provider=VoiceProvider.ELEVENLABS,
            ...     language="en-US",
            ...     gender="female"
            ... )
        """
        params = {
            "provider": provider.value if provider else None,
            "language": language,
            "gender": gender,
            "accent": accent,
            "use_case": use_case,
        }
        params = {k: v for k, v in params.items() if v is not None}
        response = self._get(Endpoints.VOICES_LIBRARY, params=params)
        return [Voice.from_dict(v) for v in response.get("voices", [])]

    def create(
        self,
        name: str,
        provider: VoiceProvider,
        provider_voice_id: str,
        description: Optional[str] = None,
        language: str = "en-US",
        config: Optional[VoiceConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Voice:
        """
        Create a custom voice configuration.

        Args:
            name: Name for the voice
            provider: Voice provider
            provider_voice_id: Voice ID from the provider
            description: Description of the voice
            language: Language code
            config: Voice configuration settings
            metadata: Custom metadata

        Returns:
            Created Voice object

        Example:
            >>> voice = client.voices.create(
            ...     name="Customer Service Voice",
            ...     provider=VoiceProvider.ELEVENLABS,
            ...     provider_voice_id="21m00Tcm4TlvDq8ikWAM",
            ...     config=VoiceConfig(stability=0.6, similarity_boost=0.8)
            ... )
        """
        data: Dict[str, Any] = {
            "name": name,
            "provider": provider.value,
            "provider_voice_id": provider_voice_id,
            "language": language,
        }

        if description:
            data["description"] = description
        if config:
            data["config"] = config.to_dict() if isinstance(config, VoiceConfig) else config
        if metadata:
            data["metadata"] = metadata

        response = self._post(Endpoints.VOICES, json=data)
        return Voice.from_dict(response)

    def update(
        self,
        voice_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[VoiceConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Voice:
        """
        Update a voice configuration.

        Args:
            voice_id: The voice's unique identifier
            name: New name
            description: New description
            config: New voice configuration
            metadata: New metadata

        Returns:
            Updated Voice object
        """
        data: Dict[str, Any] = {}

        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if config is not None:
            data["config"] = config.to_dict() if isinstance(config, VoiceConfig) else config
        if metadata is not None:
            data["metadata"] = metadata

        path = Endpoints.VOICE.format(voice_id=voice_id)
        response = self._patch(path, json=data)
        return Voice.from_dict(response)

    def delete(self, voice_id: str) -> None:
        """
        Delete a voice configuration.

        Note: You can only delete custom voices that you created.

        Args:
            voice_id: The voice's unique identifier
        """
        path = Endpoints.VOICE.format(voice_id=voice_id)
        self._delete(path)

    def preview(
        self,
        voice_id: str,
        text: str,
        output_format: str = "mp3",
    ) -> Dict[str, Any]:
        """
        Generate a preview audio sample of a voice.

        Args:
            voice_id: The voice's unique identifier
            text: Text to synthesize
            output_format: Audio format (mp3, wav, ogg)

        Returns:
            Dictionary with audio URL and metadata

        Example:
            >>> preview = client.voices.preview(
            ...     voice_id="voice_abc123",
            ...     text="Hello, this is a preview of my voice."
            ... )
            >>> print(preview["audio_url"])
        """
        path = Endpoints.VOICE_PREVIEW.format(voice_id=voice_id)
        response = self._post(path, json={
            "text": text,
            "output_format": output_format,
        })
        return response

    def clone(
        self,
        name: str,
        audio_files: List[BinaryIO],
        description: Optional[str] = None,
        provider: VoiceProvider = VoiceProvider.ELEVENLABS,
        language: str = "en-US",
        labels: Optional[Dict[str, str]] = None,
    ) -> Voice:
        """
        Clone a voice from audio samples.

        Args:
            name: Name for the cloned voice
            audio_files: List of audio file objects (at least 1 minute total)
            description: Description of the voice
            provider: Provider to use for cloning
            language: Language code
            labels: Labels for the voice (gender, age, accent, etc.)

        Returns:
            Created Voice object

        Example:
            >>> with open("sample1.mp3", "rb") as f1, open("sample2.mp3", "rb") as f2:
            ...     voice = client.voices.clone(
            ...         name="John's Voice",
            ...         audio_files=[f1, f2],
            ...         description="Cloned voice of John",
            ...         labels={"gender": "male", "age": "middle_aged"}
            ...     )
        """
        files = [("files", f) for f in audio_files]
        data = {
            "name": name,
            "provider": provider.value,
            "language": language,
        }
        if description:
            data["description"] = description
        if labels:
            data["labels"] = labels

        response = self._post(Endpoints.VOICE_CLONE, data=data, files=files)
        return Voice.from_dict(response)

    def synthesize(
        self,
        voice_id: str,
        text: str,
        output_format: str = "mp3",
        speed: float = 1.0,
        pitch: float = 1.0,
        model: Optional[str] = None,
    ) -> bytes:
        """
        Synthesize text to speech.

        Args:
            voice_id: The voice's unique identifier
            text: Text to synthesize
            output_format: Audio format (mp3, wav, ogg)
            speed: Speech speed multiplier
            pitch: Pitch adjustment
            model: TTS model to use

        Returns:
            Audio data as bytes

        Example:
            >>> audio_data = client.voices.synthesize(
            ...     voice_id="voice_abc123",
            ...     text="Hello, world!",
            ...     speed=1.0
            ... )
            >>> with open("output.mp3", "wb") as f:
            ...     f.write(audio_data)
        """
        path = f"{Endpoints.VOICE.format(voice_id=voice_id)}/synthesize"
        data = {
            "text": text,
            "output_format": output_format,
            "speed": speed,
            "pitch": pitch,
        }
        if model:
            data["model"] = model

        response = self._post(path, json=data)
        return response.get("audio", b"")

    def get_providers(self) -> List[Dict[str, Any]]:
        """
        Get list of available voice providers.

        Returns:
            List of providers with their capabilities
        """
        response = self._get(f"{Endpoints.VOICES}/providers")
        return response.get("providers", [])

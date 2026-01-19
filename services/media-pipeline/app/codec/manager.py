"""Codec manager for handling multiple audio codecs."""

from typing import Dict, Optional, Type
import structlog

from app.codec.pcmu import MuLawCodec
from app.codec.pcma import ALawCodec
from app.codec.opus import OpusCodec


logger = structlog.get_logger()


class CodecError(Exception):
    """Codec-related error."""
    pass


class BaseCodec:
    """Base class for audio codecs."""

    name: str = "base"
    sample_rate: int = 8000
    channels: int = 1
    bits_per_sample: int = 16

    def encode(self, pcm_data: bytes) -> bytes:
        """Encode PCM to codec format."""
        raise NotImplementedError

    def decode(self, encoded_data: bytes) -> bytes:
        """Decode codec format to PCM."""
        raise NotImplementedError


class PCM16Codec(BaseCodec):
    """Pass-through codec for raw PCM16."""

    name = "pcm16"
    sample_rate = 8000
    channels = 1
    bits_per_sample = 16

    def encode(self, pcm_data: bytes) -> bytes:
        return pcm_data

    def decode(self, encoded_data: bytes) -> bytes:
        return encoded_data


class CodecManager:
    """
    Manages audio codec encoding and decoding.

    Supports:
    - PCMU (G.711 μ-law) - Twilio default
    - PCMA (G.711 A-law)
    - Opus - WebRTC
    - Raw PCM16
    """

    def __init__(self):
        self._codecs: Dict[str, BaseCodec] = {
            "pcmu": MuLawCodec(),
            "pcma": ALawCodec(),
            "opus": OpusCodec(),
            "pcm16": PCM16Codec(),
            "linear16": PCM16Codec(),  # Alias
        }

        logger.info(
            "codec_manager_initialized",
            codecs=list(self._codecs.keys()),
        )

    def encode(self, pcm_data: bytes, codec_name: str) -> bytes:
        """
        Encode PCM audio to specified codec.

        Args:
            pcm_data: Raw PCM16 audio bytes
            codec_name: Target codec name

        Returns:
            Encoded audio bytes

        Raises:
            CodecError: If codec not found or encoding fails
        """
        codec = self._get_codec(codec_name)

        try:
            return codec.encode(pcm_data)
        except Exception as e:
            logger.error(
                "codec_encode_error",
                codec=codec_name,
                error=str(e),
            )
            raise CodecError(f"Failed to encode with {codec_name}: {e}")

    def decode(self, encoded_data: bytes, codec_name: str) -> bytes:
        """
        Decode audio from specified codec to PCM.

        Args:
            encoded_data: Encoded audio bytes
            codec_name: Source codec name

        Returns:
            PCM16 audio bytes

        Raises:
            CodecError: If codec not found or decoding fails
        """
        codec = self._get_codec(codec_name)

        try:
            return codec.decode(encoded_data)
        except Exception as e:
            logger.error(
                "codec_decode_error",
                codec=codec_name,
                error=str(e),
            )
            raise CodecError(f"Failed to decode with {codec_name}: {e}")

    def transcode(
        self,
        audio_data: bytes,
        from_codec: str,
        to_codec: str,
    ) -> bytes:
        """
        Transcode audio between codecs.

        Args:
            audio_data: Input audio bytes
            from_codec: Source codec name
            to_codec: Target codec name

        Returns:
            Transcoded audio bytes
        """
        if from_codec == to_codec:
            return audio_data

        # Decode to PCM
        pcm = self.decode(audio_data, from_codec)

        # Encode to target
        return self.encode(pcm, to_codec)

    def _get_codec(self, codec_name: str) -> BaseCodec:
        """Get codec by name."""
        codec_name = codec_name.lower()

        if codec_name not in self._codecs:
            raise CodecError(f"Unknown codec: {codec_name}")

        return self._codecs[codec_name]

    def register_codec(self, name: str, codec: BaseCodec) -> None:
        """Register a custom codec."""
        self._codecs[name.lower()] = codec
        logger.info("codec_registered", codec=name)

    def list_codecs(self) -> list:
        """List available codecs."""
        return list(self._codecs.keys())

    def get_codec_info(self, codec_name: str) -> dict:
        """Get codec information."""
        codec = self._get_codec(codec_name)
        return {
            "name": codec.name,
            "sample_rate": codec.sample_rate,
            "channels": codec.channels,
            "bits_per_sample": codec.bits_per_sample,
        }

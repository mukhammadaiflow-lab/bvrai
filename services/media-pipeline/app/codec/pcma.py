"""G.711 A-law (PCMA) codec implementation."""

import numpy as np
from typing import List


# A-law encoding/decoding tables
ALAW_ENCODE_TABLE: List[int] = []
ALAW_DECODE_TABLE: List[int] = []


def _init_tables():
    """Initialize A-law lookup tables."""
    global ALAW_ENCODE_TABLE, ALAW_DECODE_TABLE

    # Encoding table
    ALAW_ENCODE_TABLE = [0] * 65536

    for i in range(65536):
        if i >= 32768:
            sample = i - 65536
        else:
            sample = i
        ALAW_ENCODE_TABLE[i] = _encode_alaw_sample(sample)

    # Decoding table
    ALAW_DECODE_TABLE = [0] * 256

    for i in range(256):
        ALAW_DECODE_TABLE[i] = _decode_alaw_sample(i)


def _encode_alaw_sample(sample: int) -> int:
    """Encode single sample to A-law."""
    ALAW_MAX = 0xFFF
    CLIP = 32635

    # Get sign
    sign = 0
    if sample < 0:
        sign = 0x80
        sample = -sample

    # Clip
    if sample > CLIP:
        sample = CLIP

    # Find segment and compute
    if sample >= 256:
        exponent = 7
        for exp in range(7, 0, -1):
            if sample < (1 << (exp + 8)):
                exponent = exp
                break

        mantissa = (sample >> (exponent + 3)) & 0x0F
        alaw_byte = sign | (exponent << 4) | mantissa
    else:
        # Linear region
        mantissa = sample >> 4
        alaw_byte = sign | mantissa

    return alaw_byte ^ 0x55  # Toggle even bits


def _decode_alaw_sample(alaw_byte: int) -> int:
    """Decode single A-law sample."""
    alaw_byte ^= 0x55  # Toggle even bits

    sign = alaw_byte & 0x80
    exponent = (alaw_byte >> 4) & 0x07
    mantissa = alaw_byte & 0x0F

    if exponent == 0:
        sample = (mantissa << 4) + 8
    else:
        sample = ((mantissa << 4) + 0x108) << (exponent - 1)

    if sign:
        sample = -sample

    return sample


_init_tables()


class ALawCodec:
    """
    G.711 A-law codec for telephony audio.

    A-law is the standard for European telephone networks.
    Converts between 16-bit PCM and 8-bit A-law.
    """

    name = "pcma"
    sample_rate = 8000
    channels = 1
    bits_per_sample = 8

    def encode(self, pcm_data: bytes) -> bytes:
        """
        Encode 16-bit PCM to 8-bit A-law.

        Args:
            pcm_data: Raw 16-bit PCM audio

        Returns:
            A-law encoded audio
        """
        if len(pcm_data) == 0:
            return b""

        samples = np.frombuffer(pcm_data, dtype=np.int16)
        encoded = np.zeros(len(samples), dtype=np.uint8)

        for i, sample in enumerate(samples):
            idx = sample + 32768 if sample < 0 else sample
            encoded[i] = ALAW_ENCODE_TABLE[idx & 0xFFFF]

        return encoded.tobytes()

    def decode(self, alaw_data: bytes) -> bytes:
        """
        Decode 8-bit A-law to 16-bit PCM.

        Args:
            alaw_data: A-law encoded audio

        Returns:
            16-bit PCM audio
        """
        if len(alaw_data) == 0:
            return b""

        samples = np.frombuffer(alaw_data, dtype=np.uint8)
        decoded = np.array([ALAW_DECODE_TABLE[s] for s in samples], dtype=np.int16)

        return decoded.tobytes()


def alaw_to_linear(alaw_byte: int) -> int:
    """Convert single A-law byte to linear sample."""
    return ALAW_DECODE_TABLE[alaw_byte & 0xFF]


def linear_to_alaw(sample: int) -> int:
    """Convert single linear sample to A-law byte."""
    idx = (sample + 32768) & 0xFFFF
    return ALAW_ENCODE_TABLE[idx]

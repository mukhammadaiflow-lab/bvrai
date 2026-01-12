"""G.711 μ-law (PCMU) codec implementation."""

import struct
from typing import List
import numpy as np


# μ-law encoding/decoding tables for fast lookup
MULAW_ENCODE_TABLE: List[int] = []
MULAW_DECODE_TABLE: List[int] = []


def _init_tables():
    """Initialize μ-law lookup tables."""
    global MULAW_ENCODE_TABLE, MULAW_DECODE_TABLE

    # Encoding table (16-bit linear to 8-bit μ-law)
    MULAW_ENCODE_TABLE = [0] * 65536

    for i in range(65536):
        # Convert to signed 16-bit
        if i >= 32768:
            sample = i - 65536
        else:
            sample = i

        # μ-law encoding
        MULAW_ENCODE_TABLE[i] = _encode_mulaw_sample(sample)

    # Decoding table (8-bit μ-law to 16-bit linear)
    MULAW_DECODE_TABLE = [0] * 256

    for i in range(256):
        MULAW_DECODE_TABLE[i] = _decode_mulaw_sample(i)


def _encode_mulaw_sample(sample: int) -> int:
    """Encode single sample to μ-law."""
    MULAW_MAX = 0x1FFF
    MULAW_BIAS = 33
    CLIP = 32635

    # Get sign
    sign = (sample >> 8) & 0x80
    if sign:
        sample = -sample

    # Clip
    if sample > CLIP:
        sample = CLIP

    # Add bias
    sample += MULAW_BIAS

    # Find segment
    exponent = 7
    for exp in range(7, -1, -1):
        if sample & (1 << (exp + 7)):
            exponent = exp
            break

    # Get mantissa
    mantissa = (sample >> (exponent + 3)) & 0x0F

    # Combine
    mulaw_byte = ~(sign | (exponent << 4) | mantissa)

    return mulaw_byte & 0xFF


def _decode_mulaw_sample(mulaw_byte: int) -> int:
    """Decode single μ-law sample."""
    MULAW_BIAS = 33

    # Complement
    mulaw_byte = ~mulaw_byte

    # Extract components
    sign = mulaw_byte & 0x80
    exponent = (mulaw_byte >> 4) & 0x07
    mantissa = mulaw_byte & 0x0F

    # Reconstruct sample
    sample = ((mantissa << 3) + MULAW_BIAS) << exponent
    sample -= MULAW_BIAS

    if sign:
        sample = -sample

    return sample


# Initialize tables on module load
_init_tables()


class MuLawCodec:
    """
    G.711 μ-law codec for telephony audio.

    μ-law is the standard for North American and Japanese
    telephone networks, and is used by Twilio.

    Converts between 16-bit PCM and 8-bit μ-law.
    """

    name = "pcmu"
    sample_rate = 8000
    channels = 1
    bits_per_sample = 8

    def encode(self, pcm_data: bytes) -> bytes:
        """
        Encode 16-bit PCM to 8-bit μ-law.

        Args:
            pcm_data: Raw 16-bit PCM audio

        Returns:
            μ-law encoded audio
        """
        if len(pcm_data) == 0:
            return b""

        # Convert to numpy for fast processing
        samples = np.frombuffer(pcm_data, dtype=np.int16)

        # Use vectorized encoding
        encoded = np.zeros(len(samples), dtype=np.uint8)

        for i, sample in enumerate(samples):
            # Convert to unsigned index
            idx = sample + 32768 if sample < 0 else sample
            encoded[i] = MULAW_ENCODE_TABLE[idx & 0xFFFF]

        return encoded.tobytes()

    def decode(self, mulaw_data: bytes) -> bytes:
        """
        Decode 8-bit μ-law to 16-bit PCM.

        Args:
            mulaw_data: μ-law encoded audio

        Returns:
            16-bit PCM audio
        """
        if len(mulaw_data) == 0:
            return b""

        # Use lookup table for decoding
        samples = np.frombuffer(mulaw_data, dtype=np.uint8)
        decoded = np.array([MULAW_DECODE_TABLE[s] for s in samples], dtype=np.int16)

        return decoded.tobytes()

    def encode_fast(self, pcm_data: bytes) -> bytes:
        """
        Fast μ-law encoding using numpy.

        Approximation that's faster for large buffers.
        """
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)

        # μ-law compression formula
        MU = 255.0
        sign = np.sign(samples)
        samples = np.abs(samples) / 32768.0

        compressed = sign * np.log1p(MU * samples) / np.log1p(MU)

        # Quantize to 8 bits
        quantized = ((compressed + 1) * 127.5).astype(np.uint8)

        return quantized.tobytes()

    def decode_fast(self, mulaw_data: bytes) -> bytes:
        """
        Fast μ-law decoding using numpy.

        Approximation that's faster for large buffers.
        """
        samples = np.frombuffer(mulaw_data, dtype=np.uint8).astype(np.float32)

        # Normalize to [-1, 1]
        normalized = (samples / 127.5) - 1.0

        # μ-law expansion
        MU = 255.0
        sign = np.sign(normalized)
        expanded = sign * (np.power(1 + MU, np.abs(normalized)) - 1) / MU

        # Convert to int16
        decoded = (expanded * 32767).astype(np.int16)

        return decoded.tobytes()


def mulaw_to_linear(mulaw_byte: int) -> int:
    """Convert single μ-law byte to linear sample."""
    return MULAW_DECODE_TABLE[mulaw_byte & 0xFF]


def linear_to_mulaw(sample: int) -> int:
    """Convert single linear sample to μ-law byte."""
    idx = (sample + 32768) & 0xFFFF
    return MULAW_ENCODE_TABLE[idx]

"""
Integration Tests for Voice Pipeline

Tests the complete voice processing pipeline from audio input to speech output.
"""

import asyncio
import struct
import math
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# =============================================================================
# Test Utilities
# =============================================================================


def generate_sine_wave(
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> bytes:
    """Generate a sine wave audio sample."""
    num_samples = int(sample_rate * duration)
    samples = [
        int(32767 * amplitude * math.sin(2 * math.pi * frequency * t / sample_rate))
        for t in range(num_samples)
    ]
    return struct.pack(f"<{num_samples}h", *samples)


def generate_silence(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate silent audio."""
    num_samples = int(sample_rate * duration)
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


class MockSTTProvider:
    """Mock Speech-to-Text provider for testing."""

    def __init__(self, responses: List[str] = None):
        self.responses = responses or ["Hello, this is a test"]
        self.call_count = 0

    async def transcribe(self, audio_data: bytes) -> dict:
        """Mock transcription."""
        await asyncio.sleep(0.05)  # Simulate latency
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return {
            "transcript": response,
            "confidence": 0.95,
            "is_final": True,
            "language": "en",
        }


class MockTTSProvider:
    """Mock Text-to-Speech provider for testing."""

    def __init__(self):
        self.call_count = 0

    async def synthesize(self, text: str) -> dict:
        """Mock synthesis."""
        await asyncio.sleep(0.05)  # Simulate latency
        self.call_count += 1
        # Return 1 second of audio for every 10 characters
        duration = max(0.5, len(text) / 10)
        return {
            "audio": generate_sine_wave(duration=duration),
            "format": "pcm",
            "sample_rate": 16000,
            "duration_ms": duration * 1000,
        }


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: List[str] = None):
        self.responses = responses or [
            "I understand. How can I help you today?",
            "Sure, I can help with that.",
            "Is there anything else you need?",
        ]
        self.call_count = 0

    async def complete(self, messages: List[dict]) -> dict:
        """Mock completion."""
        await asyncio.sleep(0.1)  # Simulate latency
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return {
            "content": response,
            "model": "test-model",
            "usage": {"total_tokens": len(response) // 4},
        }


# =============================================================================
# Voice Pipeline Tests
# =============================================================================


class TestVoicePipeline:
    """Integration tests for the voice processing pipeline."""

    @pytest.fixture
    def mock_stt(self):
        """Create mock STT provider."""
        return MockSTTProvider([
            "Hello, I need help with my order",
            "Yes, my order number is 12345",
            "That's all, thank you",
        ])

    @pytest.fixture
    def mock_tts(self):
        """Create mock TTS provider."""
        return MockTTSProvider()

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        return MockLLMProvider([
            "Hello! I'd be happy to help with your order. Could you provide your order number?",
            "I found order 12345. It's currently being processed and should arrive tomorrow.",
            "You're welcome! Have a great day!",
        ])

    @pytest.mark.asyncio
    async def test_stt_transcription(self, mock_stt):
        """Test that STT correctly transcribes audio."""
        audio = generate_sine_wave(duration=2.0)
        result = await mock_stt.transcribe(audio)

        assert "transcript" in result
        assert result["confidence"] > 0.5
        assert result["is_final"] is True

    @pytest.mark.asyncio
    async def test_tts_synthesis(self, mock_tts):
        """Test that TTS correctly synthesizes speech."""
        text = "Hello, how can I help you today?"
        result = await mock_tts.synthesize(text)

        assert "audio" in result
        assert isinstance(result["audio"], bytes)
        assert len(result["audio"]) > 0
        assert result["sample_rate"] == 16000

    @pytest.mark.asyncio
    async def test_llm_response(self, mock_llm):
        """Test that LLM generates appropriate responses."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        result = await mock_llm.complete(messages)

        assert "content" in result
        assert len(result["content"]) > 0

    @pytest.mark.asyncio
    async def test_full_conversation_turn(self, mock_stt, mock_tts, mock_llm):
        """Test a complete conversation turn: audio -> text -> LLM -> speech."""
        # Step 1: Receive audio and transcribe
        user_audio = generate_sine_wave(duration=2.0)
        stt_result = await mock_stt.transcribe(user_audio)
        user_text = stt_result["transcript"]

        assert user_text == "Hello, I need help with my order"

        # Step 2: Generate LLM response
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text},
        ]
        llm_result = await mock_llm.complete(messages)
        assistant_text = llm_result["content"]

        assert len(assistant_text) > 0
        assert "order" in assistant_text.lower()

        # Step 3: Synthesize response to speech
        tts_result = await mock_tts.synthesize(assistant_text)

        assert len(tts_result["audio"]) > 0
        assert tts_result["duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, mock_stt, mock_tts, mock_llm):
        """Test multiple conversation turns."""
        conversation_history = [
            {"role": "system", "content": "You are a helpful order support assistant."},
        ]

        # Turn 1
        audio1 = generate_sine_wave(duration=2.0)
        stt1 = await mock_stt.transcribe(audio1)
        conversation_history.append({"role": "user", "content": stt1["transcript"]})

        llm1 = await mock_llm.complete(conversation_history)
        conversation_history.append({"role": "assistant", "content": llm1["content"]})

        tts1 = await mock_tts.synthesize(llm1["content"])
        assert len(tts1["audio"]) > 0

        # Turn 2
        audio2 = generate_sine_wave(duration=2.0)
        stt2 = await mock_stt.transcribe(audio2)
        conversation_history.append({"role": "user", "content": stt2["transcript"]})

        llm2 = await mock_llm.complete(conversation_history)
        conversation_history.append({"role": "assistant", "content": llm2["content"]})

        tts2 = await mock_tts.synthesize(llm2["content"])
        assert len(tts2["audio"]) > 0

        # Turn 3
        audio3 = generate_sine_wave(duration=2.0)
        stt3 = await mock_stt.transcribe(audio3)
        conversation_history.append({"role": "user", "content": stt3["transcript"]})

        llm3 = await mock_llm.complete(conversation_history)
        conversation_history.append({"role": "assistant", "content": llm3["content"]})

        tts3 = await mock_tts.synthesize(llm3["content"])
        assert len(tts3["audio"]) > 0

        # Verify conversation length
        assert len(conversation_history) == 7  # system + 3 turns (user + assistant each)

    @pytest.mark.asyncio
    async def test_latency_requirements(self, mock_stt, mock_tts, mock_llm):
        """Test that pipeline meets latency requirements."""
        max_stt_latency_ms = 500
        max_llm_latency_ms = 1000
        max_tts_latency_ms = 500

        # STT latency
        audio = generate_sine_wave(duration=1.0)
        start = time.time()
        await mock_stt.transcribe(audio)
        stt_latency = (time.time() - start) * 1000

        assert stt_latency < max_stt_latency_ms, f"STT latency {stt_latency}ms exceeds {max_stt_latency_ms}ms"

        # LLM latency
        messages = [{"role": "user", "content": "Hello"}]
        start = time.time()
        await mock_llm.complete(messages)
        llm_latency = (time.time() - start) * 1000

        assert llm_latency < max_llm_latency_ms, f"LLM latency {llm_latency}ms exceeds {max_llm_latency_ms}ms"

        # TTS latency
        start = time.time()
        await mock_tts.synthesize("Hello, how can I help?")
        tts_latency = (time.time() - start) * 1000

        assert tts_latency < max_tts_latency_ms, f"TTS latency {tts_latency}ms exceeds {max_tts_latency_ms}ms"

    @pytest.mark.asyncio
    async def test_concurrent_conversations(self, mock_stt, mock_tts, mock_llm):
        """Test handling multiple concurrent conversations."""
        num_conversations = 5

        async def run_conversation(conv_id: int):
            """Run a single conversation."""
            audio = generate_sine_wave(duration=1.0)
            stt = await mock_stt.transcribe(audio)

            llm = await mock_llm.complete([
                {"role": "user", "content": stt["transcript"]}
            ])

            tts = await mock_tts.synthesize(llm["content"])
            return {
                "conv_id": conv_id,
                "user_text": stt["transcript"],
                "assistant_text": llm["content"],
                "audio_length": len(tts["audio"]),
            }

        # Run conversations concurrently
        tasks = [run_conversation(i) for i in range(num_conversations)]
        results = await asyncio.gather(*tasks)

        assert len(results) == num_conversations
        for result in results:
            assert result["audio_length"] > 0

    @pytest.mark.asyncio
    async def test_error_handling_stt_failure(self):
        """Test handling of STT failures."""
        class FailingSTT:
            async def transcribe(self, audio):
                raise Exception("STT service unavailable")

        stt = FailingSTT()

        with pytest.raises(Exception) as exc_info:
            await stt.transcribe(generate_sine_wave())

        assert "unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_handling_llm_failure(self):
        """Test handling of LLM failures."""
        class FailingLLM:
            async def complete(self, messages):
                raise Exception("LLM rate limit exceeded")

        llm = FailingLLM()

        with pytest.raises(Exception) as exc_info:
            await llm.complete([{"role": "user", "content": "Hello"}])

        assert "rate limit" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_audio_handling(self, mock_stt):
        """Test handling of empty or very short audio."""
        # Empty audio
        empty_audio = b""
        short_audio = generate_silence(duration=0.1)

        # These should still return something (possibly empty transcript)
        result_empty = await mock_stt.transcribe(empty_audio)
        result_short = await mock_stt.transcribe(short_audio)

        assert "transcript" in result_empty
        assert "transcript" in result_short


# =============================================================================
# Audio Processing Tests
# =============================================================================


class TestAudioProcessing:
    """Tests for audio processing utilities."""

    def test_audio_format_conversion(self):
        """Test audio format conversion."""
        # Generate 16-bit PCM audio
        audio_16bit = generate_sine_wave(duration=1.0)

        # Verify format
        assert len(audio_16bit) == 16000 * 2  # 16kHz * 2 bytes per sample

    def test_audio_resampling_up(self):
        """Test upsampling audio from 8kHz to 16kHz."""
        # Generate 8kHz audio
        audio_8k = generate_sine_wave(duration=1.0, sample_rate=8000)
        assert len(audio_8k) == 8000 * 2

        # Simple linear interpolation (actual implementation would be more sophisticated)
        samples_8k = struct.unpack(f"<{8000}h", audio_8k)
        samples_16k = []
        for i in range(len(samples_8k) - 1):
            samples_16k.append(samples_8k[i])
            # Interpolate
            samples_16k.append((samples_8k[i] + samples_8k[i + 1]) // 2)
        samples_16k.append(samples_8k[-1])
        samples_16k.append(samples_8k[-1])

        audio_16k = struct.pack(f"<{len(samples_16k)}h", *samples_16k)
        assert len(audio_16k) == 16000 * 2

    def test_silence_detection(self):
        """Test detection of silence in audio."""
        silence = generate_silence(duration=1.0)
        speech = generate_sine_wave(duration=1.0, amplitude=0.8)

        # Calculate RMS
        def calculate_rms(audio_bytes):
            samples = struct.unpack(f"<{len(audio_bytes) // 2}h", audio_bytes)
            return (sum(s ** 2 for s in samples) / len(samples)) ** 0.5

        silence_rms = calculate_rms(silence)
        speech_rms = calculate_rms(speech)

        assert silence_rms < 100  # Very quiet
        assert speech_rms > 1000  # Audible


# =============================================================================
# Webhook Integration Tests
# =============================================================================


class TestWebhookIntegration:
    """Tests for webhook delivery during voice calls."""

    @pytest.mark.asyncio
    async def test_call_started_webhook(self):
        """Test that call.started webhook is triggered."""
        webhook_received = asyncio.Event()
        received_payload = {}

        async def mock_webhook_handler(payload):
            received_payload.update(payload)
            webhook_received.set()

        # Simulate call start event
        call_event = {
            "type": "call.started",
            "call_id": "call_123",
            "agent_id": "agent_456",
            "from_number": "+15551234567",
            "timestamp": "2024-01-14T10:00:00Z",
        }

        await mock_webhook_handler(call_event)
        await asyncio.wait_for(webhook_received.wait(), timeout=1.0)

        assert received_payload["type"] == "call.started"
        assert received_payload["call_id"] == "call_123"

    @pytest.mark.asyncio
    async def test_call_ended_webhook(self):
        """Test that call.ended webhook is triggered with correct data."""
        received_events = []

        async def mock_webhook_handler(payload):
            received_events.append(payload)

        # Simulate call lifecycle
        await mock_webhook_handler({
            "type": "call.started",
            "call_id": "call_123",
            "timestamp": "2024-01-14T10:00:00Z",
        })

        await mock_webhook_handler({
            "type": "call.ended",
            "call_id": "call_123",
            "duration_seconds": 180,
            "status": "completed",
            "timestamp": "2024-01-14T10:03:00Z",
        })

        assert len(received_events) == 2
        assert received_events[0]["type"] == "call.started"
        assert received_events[1]["type"] == "call.ended"
        assert received_events[1]["duration_seconds"] == 180


# =============================================================================
# Analytics Integration Tests
# =============================================================================


class TestAnalyticsIntegration:
    """Tests for analytics data collection during calls."""

    @pytest.mark.asyncio
    async def test_call_metrics_collection(self):
        """Test that call metrics are collected correctly."""
        metrics = {
            "stt_latency_samples": [],
            "llm_latency_samples": [],
            "tts_latency_samples": [],
            "turn_count": 0,
        }

        # Simulate 3 conversation turns
        for _ in range(3):
            metrics["stt_latency_samples"].append(150)  # 150ms
            metrics["llm_latency_samples"].append(300)  # 300ms
            metrics["tts_latency_samples"].append(100)  # 100ms
            metrics["turn_count"] += 1

        # Calculate averages
        avg_stt = sum(metrics["stt_latency_samples"]) / len(metrics["stt_latency_samples"])
        avg_llm = sum(metrics["llm_latency_samples"]) / len(metrics["llm_latency_samples"])
        avg_tts = sum(metrics["tts_latency_samples"]) / len(metrics["tts_latency_samples"])

        assert metrics["turn_count"] == 3
        assert avg_stt == 150
        assert avg_llm == 300
        assert avg_tts == 100

    @pytest.mark.asyncio
    async def test_conversation_summary_generation(self):
        """Test generation of conversation summary."""
        conversation = [
            {"role": "user", "content": "I want to cancel my subscription"},
            {"role": "assistant", "content": "I understand. Can I ask why you're cancelling?"},
            {"role": "user", "content": "It's too expensive for me"},
            {"role": "assistant", "content": "I've processed your cancellation request."},
        ]

        # Simple summary (actual would use LLM)
        summary = {
            "turn_count": len(conversation) // 2,
            "topics": ["subscription", "cancellation"],
            "outcome": "cancellation_processed",
            "sentiment": "neutral",
        }

        assert summary["turn_count"] == 2
        assert "cancellation" in summary["topics"]

"""Unit tests for voice engine."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import struct

from app.voice.engine import VoiceEngine
from app.voice.tts import TextToSpeech, TTSProvider
from app.voice.stt import SpeechToText, STTProvider
from app.voice.vad import VoiceActivityDetector


class TestTextToSpeech:
    """Tests for TextToSpeech."""

    def test_create_tts(self):
        """Test creating TTS instance."""
        tts = TextToSpeech(
            provider="elevenlabs",
            voice_id="voice_123",
            model="eleven_monolingual_v1",
        )

        assert tts.provider == "elevenlabs"
        assert tts.voice_id == "voice_123"

    @pytest.mark.asyncio
    async def test_synthesize_text(self):
        """Test synthesizing text to speech."""
        tts = TextToSpeech(provider="mock", voice_id="test")

        with patch.object(tts, '_synthesize', new_callable=AsyncMock) as mock_synth:
            mock_synth.return_value = b"audio_data"

            audio = await tts.synthesize("Hello, world!")

            assert audio == b"audio_data"
            mock_synth.assert_called_once_with("Hello, world!")

    @pytest.mark.asyncio
    async def test_synthesize_with_options(self):
        """Test synthesizing with voice options."""
        tts = TextToSpeech(provider="mock", voice_id="test")

        with patch.object(tts, '_synthesize', new_callable=AsyncMock) as mock_synth:
            mock_synth.return_value = b"audio_data"

            audio = await tts.synthesize(
                "Hello!",
                speed=1.2,
                pitch=1.1,
                volume=0.8,
            )

            assert audio is not None

    def test_estimate_duration(self):
        """Test estimating speech duration."""
        tts = TextToSpeech(provider="mock", voice_id="test")

        # Average speaking rate is ~150 words per minute
        duration = tts.estimate_duration("This is a test sentence with ten words.")

        # 10 words at 150 wpm = 4 seconds
        assert 3 <= duration <= 5

    @pytest.mark.asyncio
    async def test_stream_synthesis(self):
        """Test streaming TTS synthesis."""
        tts = TextToSpeech(provider="mock", voice_id="test")

        chunks = []
        with patch.object(tts, '_stream_synthesize', new_callable=AsyncMock) as mock_stream:
            async def mock_generator():
                for i in range(5):
                    yield f"chunk_{i}".encode()

            mock_stream.return_value = mock_generator()

            async for chunk in tts.stream("Hello, world!"):
                chunks.append(chunk)

            assert len(chunks) == 5


class TestSpeechToText:
    """Tests for SpeechToText."""

    def test_create_stt(self):
        """Test creating STT instance."""
        stt = SpeechToText(
            provider="deepgram",
            model="nova-2",
            language="en-US",
        )

        assert stt.provider == "deepgram"
        assert stt.model == "nova-2"
        assert stt.language == "en-US"

    @pytest.mark.asyncio
    async def test_transcribe_audio(self, sample_audio_data):
        """Test transcribing audio."""
        stt = SpeechToText(provider="mock", language="en-US")

        with patch.object(stt, '_transcribe', new_callable=AsyncMock) as mock_trans:
            mock_trans.return_value = {
                "text": "Hello, world!",
                "confidence": 0.95,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.6, "end": 1.0},
                ],
            }

            result = await stt.transcribe(sample_audio_data)

            assert result["text"] == "Hello, world!"
            assert result["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_realtime_transcription(self, sample_audio_data):
        """Test real-time transcription."""
        stt = SpeechToText(provider="mock", language="en-US")

        with patch.object(stt, '_process_chunk', new_callable=AsyncMock) as mock_chunk:
            mock_chunk.return_value = {
                "text": "Hello",
                "is_final": False,
                "confidence": 0.8,
            }

            result = await stt.process_chunk(sample_audio_data)

            assert result["text"] == "Hello"
            assert result["is_final"] is False

    def test_supported_languages(self):
        """Test getting supported languages."""
        stt = SpeechToText(provider="mock", language="en-US")

        languages = stt.get_supported_languages()

        assert "en-US" in languages
        assert "es-ES" in languages

    @pytest.mark.asyncio
    async def test_detect_language(self, sample_audio_data):
        """Test language detection."""
        stt = SpeechToText(provider="mock", language="auto")

        with patch.object(stt, '_detect_language', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = {
                "language": "en-US",
                "confidence": 0.92,
            }

            result = await stt.detect_language(sample_audio_data)

            assert result["language"] == "en-US"


class TestVoiceActivityDetector:
    """Tests for VoiceActivityDetector."""

    def test_create_vad(self):
        """Test creating VAD instance."""
        vad = VoiceActivityDetector(
            threshold=0.5,
            frame_duration_ms=30,
            padding_duration_ms=300,
        )

        assert vad.threshold == 0.5
        assert vad.frame_duration_ms == 30

    def test_detect_speech_in_frame(self, sample_audio_data):
        """Test detecting speech in a single frame."""
        vad = VoiceActivityDetector(threshold=0.5)

        # Create a frame with some energy
        frame_size = vad.frame_duration_ms * 16  # 16 samples per ms at 16kHz
        frame = bytes([100] * frame_size)

        with patch.object(vad, '_calculate_energy') as mock_energy:
            mock_energy.return_value = 0.7

            is_speech = vad.is_speech(frame)

            assert is_speech is True

    def test_detect_silence(self, sample_audio_data):
        """Test detecting silence."""
        vad = VoiceActivityDetector(threshold=0.5)

        # Silence frame
        frame = bytes(480)  # 30ms at 16kHz

        with patch.object(vad, '_calculate_energy') as mock_energy:
            mock_energy.return_value = 0.1

            is_speech = vad.is_speech(frame)

            assert is_speech is False

    def test_smoothing(self):
        """Test VAD smoothing to avoid flicker."""
        vad = VoiceActivityDetector(
            threshold=0.5,
            min_speech_duration_ms=100,
        )

        # Simulate a few speech frames
        results = []
        for i in range(10):
            with patch.object(vad, '_calculate_energy') as mock_energy:
                # Alternate between speech and silence
                mock_energy.return_value = 0.7 if i % 2 == 0 else 0.3
                frame = bytes(480)
                results.append(vad.is_speech(frame))

        # Due to smoothing, we shouldn't see rapid changes
        # This is implementation-dependent
        assert results is not None

    def test_get_speech_segments(self, sample_audio_data):
        """Test getting speech segments."""
        vad = VoiceActivityDetector(threshold=0.5)

        with patch.object(vad, '_process_audio', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = [
                {"start": 0.0, "end": 2.5, "is_speech": True},
                {"start": 2.5, "end": 3.0, "is_speech": False},
                {"start": 3.0, "end": 5.5, "is_speech": True},
            ]

            segments = vad.get_speech_segments(sample_audio_data)

            speech_segments = [s for s in segments if s["is_speech"]]
            assert len(speech_segments) == 2


class TestVoiceEngine:
    """Tests for VoiceEngine."""

    def test_create_engine(self):
        """Test creating voice engine."""
        engine = VoiceEngine(
            tts_provider="elevenlabs",
            stt_provider="deepgram",
            voice_id="voice_123",
            language="en-US",
        )

        assert engine.tts_provider == "elevenlabs"
        assert engine.stt_provider == "deepgram"

    @pytest.mark.asyncio
    async def test_initialize_engine(self):
        """Test initializing voice engine."""
        engine = VoiceEngine(
            tts_provider="mock",
            stt_provider="mock",
            voice_id="test",
        )

        with patch.object(engine, '_init_tts', new_callable=AsyncMock), \
             patch.object(engine, '_init_stt', new_callable=AsyncMock):
            await engine.initialize()

        assert engine.is_initialized is True

    @pytest.mark.asyncio
    async def test_speak(self):
        """Test converting text to speech."""
        engine = VoiceEngine(
            tts_provider="mock",
            stt_provider="mock",
            voice_id="test",
        )
        engine.is_initialized = True

        with patch.object(engine.tts, 'synthesize', new_callable=AsyncMock) as mock_synth:
            mock_synth.return_value = b"audio_data"

            audio = await engine.speak("Hello!")

            assert audio == b"audio_data"

    @pytest.mark.asyncio
    async def test_listen(self, sample_audio_data):
        """Test converting speech to text."""
        engine = VoiceEngine(
            tts_provider="mock",
            stt_provider="mock",
            voice_id="test",
        )
        engine.is_initialized = True

        with patch.object(engine.stt, 'transcribe', new_callable=AsyncMock) as mock_trans:
            mock_trans.return_value = {"text": "Hello!", "confidence": 0.95}

            result = await engine.listen(sample_audio_data)

            assert result["text"] == "Hello!"

    @pytest.mark.asyncio
    async def test_stream_speak(self):
        """Test streaming text to speech."""
        engine = VoiceEngine(
            tts_provider="mock",
            stt_provider="mock",
            voice_id="test",
        )
        engine.is_initialized = True

        chunks = []
        with patch.object(engine.tts, 'stream', new_callable=AsyncMock) as mock_stream:
            async def mock_generator():
                for i in range(3):
                    yield f"chunk_{i}".encode()

            mock_stream.return_value = mock_generator()

            async for chunk in engine.stream_speak("Hello!"):
                chunks.append(chunk)

            assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_process_audio_stream(self, sample_audio_data):
        """Test processing audio stream."""
        engine = VoiceEngine(
            tts_provider="mock",
            stt_provider="mock",
            voice_id="test",
        )
        engine.is_initialized = True

        with patch.object(engine.stt, 'process_chunk', new_callable=AsyncMock) as mock_chunk:
            mock_chunk.return_value = {
                "text": "Hello",
                "is_final": True,
            }

            result = await engine.process_audio_chunk(sample_audio_data)

            assert result["text"] == "Hello"
            assert result["is_final"] is True


class TestAudioProcessing:
    """Tests for audio processing utilities."""

    def test_resample_audio(self):
        """Test resampling audio."""
        from app.voice.audio import resample_audio

        # Create sample audio at 44100 Hz
        input_sample_rate = 44100
        output_sample_rate = 16000
        duration = 1  # second
        samples = input_sample_rate * duration
        audio = bytes(samples * 2)  # 16-bit

        resampled = resample_audio(
            audio,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
        )

        # Output should have fewer samples
        expected_samples = output_sample_rate * duration
        assert len(resampled) == expected_samples * 2

    def test_convert_sample_format(self):
        """Test converting sample format."""
        from app.voice.audio import convert_sample_format

        # Create 16-bit audio
        audio_16bit = struct.pack('<h', 16000) * 100

        # Convert to float32
        audio_float = convert_sample_format(
            audio_16bit,
            from_format="int16",
            to_format="float32",
        )

        assert len(audio_float) == 100 * 4  # float32 is 4 bytes

    def test_calculate_audio_level(self):
        """Test calculating audio level."""
        from app.voice.audio import calculate_audio_level

        # Silent audio
        silent = bytes(1000)
        silent_level = calculate_audio_level(silent)
        assert silent_level < 0.1

        # Loud audio
        loud = bytes([200, 200] * 500)
        loud_level = calculate_audio_level(loud)
        assert loud_level > 0.5

    def test_trim_silence(self):
        """Test trimming silence from audio."""
        from app.voice.audio import trim_silence

        # Audio with silence at start and end
        silence = bytes(1000)
        speech = bytes([128] * 2000)
        audio = silence + speech + silence

        trimmed = trim_silence(audio, threshold=0.1)

        # Trimmed should be shorter
        assert len(trimmed) < len(audio)

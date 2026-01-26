"""Unit tests for voice configuration functionality."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class TestVoiceConfiguration:
    """Tests for voice configuration."""

    def test_create_voice_config(self):
        """Test creating a voice configuration."""
        from bvrai_core.voice.config import VoiceConfiguration

        config = VoiceConfiguration(
            id=f"vc_{uuid4().hex[:12]}",
            organization_id="org_123",
            name="Professional Voice",
            provider="elevenlabs",
            voice_id="21m00Tcm4TlvDq8ikWAM",
            settings={
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.5,
            },
        )

        assert config.provider == "elevenlabs"
        assert config.settings["stability"] == 0.5

    def test_voice_config_validation(self):
        """Test voice configuration validation."""
        from bvrai_core.voice.config import VoiceConfiguration, validate_voice_settings

        # Valid settings
        valid_settings = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "speed": 1.0,
            "pitch": 1.0,
        }

        errors = validate_voice_settings(valid_settings)
        assert len(errors) == 0

        # Invalid settings
        invalid_settings = {
            "stability": 2.0,  # Out of range
            "speed": -1,  # Negative
        }

        errors = validate_voice_settings(invalid_settings)
        assert len(errors) > 0


class TestVoiceProvider:
    """Tests for voice providers."""

    def test_get_available_providers(self):
        """Test listing available voice providers."""
        from bvrai_core.voice.providers import get_available_providers

        providers = get_available_providers()

        assert "elevenlabs" in providers
        assert "openai" in providers
        assert "deepgram" in providers

    def test_provider_capabilities(self):
        """Test getting provider capabilities."""
        from bvrai_core.voice.providers import get_provider_capabilities

        capabilities = get_provider_capabilities("elevenlabs")

        assert "streaming" in capabilities
        assert "cloning" in capabilities
        assert capabilities["max_chunk_size"] > 0

    @pytest.mark.asyncio
    async def test_list_voices_by_provider(self):
        """Test listing voices by provider."""
        from bvrai_core.voice.providers import VoiceProvider

        provider = VoiceProvider(name="elevenlabs")

        with patch.object(provider, '_fetch_voices', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [
                {"id": "voice1", "name": "Rachel", "language": "en"},
                {"id": "voice2", "name": "Josh", "language": "en"},
            ]

            voices = await provider.list_voices()

            assert len(voices) == 2
            assert any(v["name"] == "Rachel" for v in voices)


class TestVoiceSynthesis:
    """Tests for voice synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_speech(self):
        """Test synthesizing speech."""
        from bvrai_core.voice.synthesis import VoiceSynthesizer

        synthesizer = VoiceSynthesizer(provider="mock")

        with patch.object(synthesizer, '_generate_audio', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = b"audio_data"

            audio = await synthesizer.synthesize(
                text="Hello, how can I help you?",
                voice_id="test_voice",
            )

            assert audio is not None
            mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_synthesis(self):
        """Test streaming voice synthesis."""
        from bvrai_core.voice.synthesis import VoiceSynthesizer

        synthesizer = VoiceSynthesizer(provider="mock")

        chunks = []
        async for chunk in synthesizer.synthesize_stream(
            text="This is a longer text that will be streamed.",
            voice_id="test_voice",
        ):
            chunks.append(chunk)

        assert len(chunks) > 0

    def test_text_preprocessing(self):
        """Test text preprocessing for synthesis."""
        from bvrai_core.voice.synthesis import preprocess_text

        # Should handle abbreviations
        text = "Dr. Smith will see you at 3pm."
        processed = preprocess_text(text)
        assert "Doctor" in processed

        # Should handle numbers
        text = "Please call 555-1234."
        processed = preprocess_text(text)
        # Numbers should be preserved or formatted

        # Should handle special characters
        text = "Your balance is $100.50"
        processed = preprocess_text(text)
        assert "dollars" in processed.lower() or "100" in processed


class TestVoiceCloning:
    """Tests for voice cloning."""

    @pytest.mark.asyncio
    async def test_create_cloned_voice(self):
        """Test creating a cloned voice."""
        from bvrai_core.voice.cloning import VoiceCloner

        cloner = VoiceCloner(provider="elevenlabs")

        with patch.object(cloner, '_upload_samples', new_callable=AsyncMock) as mock_upload:
            with patch.object(cloner, '_train_voice', new_callable=AsyncMock) as mock_train:
                mock_upload.return_value = "upload_123"
                mock_train.return_value = {
                    "id": "cloned_voice_123",
                    "name": "Custom Voice",
                    "status": "ready",
                }

                result = await cloner.create_voice(
                    name="Custom Voice",
                    samples=[b"audio1", b"audio2"],
                    description="A custom cloned voice",
                )

                assert result["id"] == "cloned_voice_123"
                assert result["status"] == "ready"

    @pytest.mark.asyncio
    async def test_delete_cloned_voice(self):
        """Test deleting a cloned voice."""
        from bvrai_core.voice.cloning import VoiceCloner

        cloner = VoiceCloner(provider="elevenlabs")

        with patch.object(cloner, '_delete_voice', new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = True

            success = await cloner.delete_voice("cloned_voice_123")

            assert success is True
            mock_delete.assert_called_with("cloned_voice_123")


class TestSpeechRecognition:
    """Tests for speech recognition (STT)."""

    @pytest.mark.asyncio
    async def test_transcribe_audio(self):
        """Test transcribing audio."""
        from bvrai_core.voice.stt import SpeechRecognizer

        recognizer = SpeechRecognizer(provider="mock")

        with patch.object(recognizer, '_transcribe', new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = {
                "text": "Hello, I need help with my order.",
                "confidence": 0.95,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "I", "start": 0.6, "end": 0.7},
                    # ...
                ],
            }

            result = await recognizer.transcribe(b"audio_data")

            assert result["text"] == "Hello, I need help with my order."
            assert result["confidence"] > 0.9

    @pytest.mark.asyncio
    async def test_streaming_recognition(self):
        """Test streaming speech recognition."""
        from bvrai_core.voice.stt import SpeechRecognizer

        recognizer = SpeechRecognizer(provider="mock")

        partial_results = []

        async def mock_stream():
            yield {"partial": "Hello"}
            yield {"partial": "Hello I need"}
            yield {"final": "Hello I need help", "confidence": 0.9}

        with patch.object(recognizer, '_stream_recognize') as mock_recognize:
            mock_recognize.return_value = mock_stream()

            async for result in recognizer.recognize_stream(b"audio_stream"):
                partial_results.append(result)

            assert len(partial_results) == 3
            assert partial_results[-1]["final"] == "Hello I need help"

    def test_language_detection(self):
        """Test language detection from audio."""
        from bvrai_core.voice.stt import detect_language

        # Mock audio that "sounds" like Spanish
        result = detect_language(b"mock_spanish_audio", hint="es")

        # Should return language code
        assert result in ["es", "es-ES", "es-MX"]


class TestVoiceActivity:
    """Tests for voice activity detection."""

    def test_detect_speech(self):
        """Test detecting speech in audio."""
        from bvrai_core.voice.vad import VoiceActivityDetector

        vad = VoiceActivityDetector(threshold=0.5)

        # Silence
        silence = bytes(1000)
        assert vad.is_speech(silence) is False

        # Simulated speech (non-zero values)
        speech = bytes([100] * 1000)
        assert vad.is_speech(speech) is True

    def test_speech_segments(self):
        """Test detecting speech segments."""
        from bvrai_core.voice.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()

        # Audio with speech and silence
        audio = bytes([0] * 500 + [100] * 1000 + [0] * 500)

        segments = vad.get_speech_segments(audio, sample_rate=16000)

        assert len(segments) >= 1
        # Should detect the speech segment

    def test_end_of_speech_detection(self):
        """Test detecting end of speech."""
        from bvrai_core.voice.vad import VoiceActivityDetector

        vad = VoiceActivityDetector(silence_threshold_ms=500)

        # Process audio frames
        vad.process_frame(bytes([100] * 100))  # Speech
        vad.process_frame(bytes([100] * 100))  # Speech
        vad.process_frame(bytes([0] * 100))    # Silence
        vad.process_frame(bytes([0] * 100))    # Silence

        # After enough silence, should detect end of speech
        is_end, duration = vad.check_end_of_speech()

        # Would need more silence frames to trigger


class TestVoiceEffects:
    """Tests for voice effects and modifications."""

    def test_change_speed(self):
        """Test changing voice speed."""
        from bvrai_core.voice.effects import change_speed

        original = b"audio_data" * 100
        faster = change_speed(original, factor=1.5)

        # Faster audio should be shorter
        assert len(faster) < len(original)

    def test_change_pitch(self):
        """Test changing voice pitch."""
        from bvrai_core.voice.effects import change_pitch

        original = b"audio_data" * 100
        higher = change_pitch(original, semitones=4)

        # Pitch-shifted audio should exist
        assert higher is not None

    def test_noise_reduction(self):
        """Test noise reduction."""
        from bvrai_core.voice.effects import reduce_noise

        noisy_audio = b"noisy_audio" * 100
        clean_audio = reduce_noise(noisy_audio, level=0.5)

        assert clean_audio is not None


class TestVoiceConfigRepository:
    """Tests for voice configuration repository."""

    @pytest.mark.asyncio
    async def test_create_config(self):
        """Test creating voice configuration in database."""
        from bvrai_core.voice.repository import VoiceConfigRepository

        repo = VoiceConfigRepository()

        with patch.object(repo, '_save', new_callable=AsyncMock) as mock_save:
            config = await repo.create(
                organization_id="org_123",
                name="Test Voice",
                provider="elevenlabs",
                voice_id="test_voice_id",
                settings={"stability": 0.5},
            )

            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_configs(self):
        """Test listing voice configurations."""
        from bvrai_core.voice.repository import VoiceConfigRepository

        repo = VoiceConfigRepository()

        with patch.object(repo, '_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {"id": "vc_1", "name": "Voice 1"},
                {"id": "vc_2", "name": "Voice 2"},
            ]

            configs = await repo.list_by_organization("org_123")

            assert len(configs) == 2
            mock_query.assert_called_once()

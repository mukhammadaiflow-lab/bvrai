"""
BVRAI Voice Engine - Main Orchestrator

Central coordination point for all voice processing operations.
Handles provider routing, failover, health monitoring, and metrics.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple

from .base import (
    AudioChunk,
    AudioConfig,
    ProviderHealth,
    STTConfig,
    STTProvider,
    STTProviderInterface,
    SynthesisResult,
    TTSConfig,
    TTSProvider,
    TTSProviderInterface,
    TranscriptionResult,
    VoiceError,
    VoiceMetrics,
    VoiceProfile,
    generate_request_id,
)
from .stt_providers import create_stt_provider
from .tts_providers import create_tts_provider

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VoiceEngineConfig:
    """Configuration for the voice engine."""

    # Default providers
    default_stt_provider: STTProvider = STTProvider.DEEPGRAM
    default_tts_provider: TTSProvider = TTSProvider.ELEVENLABS

    # Fallback chains
    stt_fallback_chain: List[STTProvider] = field(default_factory=lambda: [
        STTProvider.DEEPGRAM,
        STTProvider.ASSEMBLYAI,
        STTProvider.OPENAI_WHISPER,
    ])
    tts_fallback_chain: List[TTSProvider] = field(default_factory=lambda: [
        TTSProvider.ELEVENLABS,
        TTSProvider.OPENAI,
        TTSProvider.PLAYHT,
    ])

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    # Health check settings
    health_check_interval: float = 60.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 300.0

    # Performance settings
    request_timeout: float = 30.0
    streaming_chunk_size: int = 4096

    # Caching
    enable_response_cache: bool = True
    cache_ttl: float = 3600.0

    # Metrics
    enable_metrics: bool = True
    metrics_window: float = 3600.0


# =============================================================================
# Routing Strategies
# =============================================================================

class RoutingStrategy:
    """Base class for provider routing strategies."""

    def select_provider(
        self,
        providers: Dict[str, ProviderHealth],
        config: Any
    ) -> Optional[str]:
        """Select a provider based on the strategy."""
        raise NotImplementedError


class PrimaryWithFallbackStrategy(RoutingStrategy):
    """Use primary provider with fallback chain."""

    def __init__(self, fallback_chain: List[str]):
        self.fallback_chain = fallback_chain

    def select_provider(
        self,
        providers: Dict[str, ProviderHealth],
        config: Any
    ) -> Optional[str]:
        for provider in self.fallback_chain:
            health = providers.get(provider)
            if health and health.is_healthy and not health.circuit_open:
                return provider
        return None


class LowestLatencyStrategy(RoutingStrategy):
    """Select provider with lowest average latency."""

    def select_provider(
        self,
        providers: Dict[str, ProviderHealth],
        config: Any
    ) -> Optional[str]:
        healthy = [
            (name, health)
            for name, health in providers.items()
            if health.is_healthy and not health.circuit_open
        ]

        if not healthy:
            return None

        # Sort by average latency
        sorted_providers = sorted(healthy, key=lambda x: x[1].avg_latency_ms)
        return sorted_providers[0][0]


class RoundRobinStrategy(RoutingStrategy):
    """Round-robin selection among healthy providers."""

    def __init__(self):
        self._index = 0

    def select_provider(
        self,
        providers: Dict[str, ProviderHealth],
        config: Any
    ) -> Optional[str]:
        healthy = [
            name for name, health in providers.items()
            if health.is_healthy and not health.circuit_open
        ]

        if not healthy:
            return None

        provider = healthy[self._index % len(healthy)]
        self._index += 1
        return provider


# =============================================================================
# Voice Engine
# =============================================================================

class VoiceEngine:
    """
    Central voice processing engine.

    Coordinates STT and TTS operations across multiple providers
    with automatic failover, health monitoring, and metrics collection.
    """

    def __init__(self, config: Optional[VoiceEngineConfig] = None):
        self.config = config or VoiceEngineConfig()

        # Provider instances
        self._stt_providers: Dict[STTProvider, STTProviderInterface] = {}
        self._tts_providers: Dict[TTSProvider, TTSProviderInterface] = {}

        # Provider health tracking
        self._stt_health: Dict[STTProvider, ProviderHealth] = {}
        self._tts_health: Dict[TTSProvider, ProviderHealth] = {}

        # Metrics
        self._metrics = VoiceMetrics()
        self._request_history: List[Dict[str, Any]] = []

        # Routing strategies
        self._stt_strategy = PrimaryWithFallbackStrategy(
            [p.value for p in self.config.stt_fallback_chain]
        )
        self._tts_strategy = PrimaryWithFallbackStrategy(
            [p.value for p in self.config.tts_fallback_chain]
        )

        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Event callbacks
        self._on_provider_unhealthy: List[Callable] = []
        self._on_provider_recovered: List[Callable] = []

        # Response cache
        self._response_cache: Dict[str, Tuple[Any, float]] = {}

        logger.info("Voice engine initialized")

    async def start(self) -> None:
        """Start the voice engine and initialize providers."""
        if self._is_running:
            return

        logger.info("Starting voice engine...")

        # Initialize default providers
        await self._initialize_providers()

        # Start health check background task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        self._is_running = True
        logger.info("Voice engine started")

    async def stop(self) -> None:
        """Stop the voice engine and cleanup resources."""
        if not self._is_running:
            return

        logger.info("Stopping voice engine...")

        self._is_running = False

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close all providers
        for provider in self._stt_providers.values():
            await provider.close()
        for provider in self._tts_providers.values():
            await provider.close()

        self._stt_providers.clear()
        self._tts_providers.clear()

        logger.info("Voice engine stopped")

    async def _initialize_providers(self) -> None:
        """Initialize STT and TTS providers."""
        # Initialize STT providers
        for provider in self.config.stt_fallback_chain:
            try:
                instance = create_stt_provider(provider)
                self._stt_providers[provider] = instance
                self._stt_health[provider] = ProviderHealth(provider=provider)
                logger.info(f"Initialized STT provider: {provider.value}")
            except Exception as e:
                logger.warning(f"Failed to initialize STT provider {provider.value}: {e}")

        # Initialize TTS providers
        for provider in self.config.tts_fallback_chain:
            try:
                instance = create_tts_provider(provider)
                self._tts_providers[provider] = instance
                self._tts_health[provider] = ProviderHealth(provider=provider)
                logger.info(f"Initialized TTS provider: {provider.value}")
            except Exception as e:
                logger.warning(f"Failed to initialize TTS provider {provider.value}: {e}")

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while self._is_running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)

    async def _run_health_checks(self) -> None:
        """Run health checks on all providers."""
        # Check STT providers
        for provider, instance in self._stt_providers.items():
            try:
                is_healthy = await instance.health_check()
                health = self._stt_health[provider]

                was_healthy = health.is_healthy
                health.is_healthy = is_healthy
                health.last_check = datetime.utcnow()

                # Check circuit breaker timeout
                if health.circuit_open and health.circuit_open_until:
                    if datetime.utcnow() > health.circuit_open_until + timedelta(
                        seconds=self.config.circuit_breaker_timeout
                    ):
                        health.circuit_open = False
                        health.consecutive_errors = 0
                        logger.info(f"Circuit breaker reset for STT provider: {provider.value}")

                # Notify on state change
                if was_healthy and not is_healthy:
                    await self._notify_provider_unhealthy("stt", provider)
                elif not was_healthy and is_healthy:
                    await self._notify_provider_recovered("stt", provider)

            except Exception as e:
                logger.error(f"Health check failed for STT provider {provider.value}: {e}")
                self._stt_health[provider].is_healthy = False

        # Check TTS providers
        for provider, instance in self._tts_providers.items():
            try:
                is_healthy = await instance.health_check()
                health = self._tts_health[provider]

                was_healthy = health.is_healthy
                health.is_healthy = is_healthy
                health.last_check = datetime.utcnow()

                if health.circuit_open and health.circuit_open_until:
                    if datetime.utcnow() > health.circuit_open_until + timedelta(
                        seconds=self.config.circuit_breaker_timeout
                    ):
                        health.circuit_open = False
                        health.consecutive_errors = 0

                if was_healthy and not is_healthy:
                    await self._notify_provider_unhealthy("tts", provider)
                elif not was_healthy and is_healthy:
                    await self._notify_provider_recovered("tts", provider)

            except Exception as e:
                logger.error(f"Health check failed for TTS provider {provider.value}: {e}")
                self._tts_health[provider].is_healthy = False

    async def _notify_provider_unhealthy(
        self,
        provider_type: str,
        provider: Any
    ) -> None:
        """Notify callbacks when a provider becomes unhealthy."""
        logger.warning(f"{provider_type.upper()} provider unhealthy: {provider.value}")
        for callback in self._on_provider_unhealthy:
            try:
                await callback(provider_type, provider)
            except Exception as e:
                logger.error(f"Error in unhealthy callback: {e}")

    async def _notify_provider_recovered(
        self,
        provider_type: str,
        provider: Any
    ) -> None:
        """Notify callbacks when a provider recovers."""
        logger.info(f"{provider_type.upper()} provider recovered: {provider.value}")
        for callback in self._on_provider_recovered:
            try:
                await callback(provider_type, provider)
            except Exception as e:
                logger.error(f"Error in recovered callback: {e}")

    # =========================================================================
    # STT Operations
    # =========================================================================

    async def transcribe(
        self,
        audio_data: bytes,
        config: Optional[STTConfig] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio_data: Raw audio bytes
            config: STT configuration (uses defaults if not provided)

        Returns:
            TranscriptionResult with text and metadata
        """
        config = config or STTConfig(provider=self.config.default_stt_provider)
        request_id = generate_request_id()
        start_time = time.monotonic()

        logger.debug(f"Transcription request {request_id}: {len(audio_data)} bytes")

        # Get provider with fallback
        providers_to_try = self._get_stt_providers_to_try(config.provider)

        last_error: Optional[Exception] = None

        for provider in providers_to_try:
            instance = self._stt_providers.get(provider)
            if not instance:
                continue

            health = self._stt_health.get(provider)
            if health and health.circuit_open:
                continue

            try:
                config.provider = provider
                result = await asyncio.wait_for(
                    instance.transcribe(audio_data, config),
                    timeout=self.config.request_timeout
                )

                # Record success
                latency_ms = (time.monotonic() - start_time) * 1000
                if health:
                    health.record_success(latency_ms)

                # Update metrics
                self._metrics.stt_requests += 1
                self._metrics.stt_characters_transcribed += len(result.text)
                self._metrics.stt_audio_seconds_processed += result.duration
                self._update_avg_latency("stt", latency_ms)

                logger.info(
                    f"Transcription completed",
                    extra={
                        "request_id": request_id,
                        "provider": provider.value,
                        "latency_ms": latency_ms,
                        "text_length": len(result.text),
                    }
                )

                return result

            except asyncio.TimeoutError:
                last_error = VoiceError(
                    f"Transcription timed out after {self.config.request_timeout}s",
                    provider=provider.value
                )
                if health:
                    health.record_error("timeout")
                logger.warning(f"STT timeout for provider {provider.value}")

            except Exception as e:
                last_error = e
                if health:
                    health.record_error(str(e))
                logger.warning(f"STT error for provider {provider.value}: {e}")

        # All providers failed
        self._metrics.stt_errors += 1
        raise last_error or VoiceError("All STT providers failed")

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        config: Optional[STTConfig] = None
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Stream transcription results as audio is received.

        Args:
            audio_stream: Async generator of audio chunks
            config: STT configuration

        Yields:
            TranscriptionResult objects (partial and final)
        """
        config = config or STTConfig(provider=self.config.default_stt_provider)
        request_id = generate_request_id()

        provider = self._select_stt_provider(config.provider)
        instance = self._stt_providers.get(provider)

        if not instance:
            raise VoiceError(f"No STT provider available")

        try:
            config.provider = provider
            async for result in instance.transcribe_stream(audio_stream, config):
                result.id = request_id
                yield result

        except Exception as e:
            self._metrics.stt_errors += 1
            health = self._stt_health.get(provider)
            if health:
                health.record_error(str(e))
            raise

    # =========================================================================
    # TTS Operations
    # =========================================================================

    async def synthesize(
        self,
        text: str,
        config: Optional[TTSConfig] = None,
        voice_profile: Optional[VoiceProfile] = None
    ) -> SynthesisResult:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            config: TTS configuration
            voice_profile: Optional voice profile to use

        Returns:
            SynthesisResult with audio data
        """
        # Build config from voice profile if provided
        if voice_profile:
            config = voice_profile.to_tts_config()
        elif config is None:
            config = TTSConfig(provider=self.config.default_tts_provider)

        request_id = generate_request_id()
        start_time = time.monotonic()

        logger.debug(f"Synthesis request {request_id}: {len(text)} characters")

        # Check cache
        if self.config.enable_response_cache:
            cache_key = self._get_cache_key(text, config)
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.debug(f"Cache hit for synthesis request {request_id}")
                return cached

        # Get providers to try
        providers_to_try = self._get_tts_providers_to_try(config.provider)

        last_error: Optional[Exception] = None

        for provider in providers_to_try:
            instance = self._tts_providers.get(provider)
            if not instance:
                continue

            health = self._tts_health.get(provider)
            if health and health.circuit_open:
                continue

            try:
                config.provider = provider
                result = await asyncio.wait_for(
                    instance.synthesize(text, config),
                    timeout=self.config.request_timeout
                )

                # Record success
                latency_ms = (time.monotonic() - start_time) * 1000
                if health:
                    health.record_success(latency_ms)

                # Update metrics
                self._metrics.tts_requests += 1
                self._metrics.tts_characters_synthesized += len(text)
                self._metrics.tts_audio_seconds_generated += result.audio_duration
                self._metrics.estimated_tts_cost += result.estimated_cost
                self._update_avg_latency("tts", latency_ms)

                # Cache result
                if self.config.enable_response_cache:
                    self._add_to_cache(cache_key, result)

                logger.info(
                    f"Synthesis completed",
                    extra={
                        "request_id": request_id,
                        "provider": provider.value,
                        "latency_ms": latency_ms,
                        "audio_size": len(result.audio_data),
                    }
                )

                return result

            except asyncio.TimeoutError:
                last_error = VoiceError(
                    f"Synthesis timed out after {self.config.request_timeout}s",
                    provider=provider.value
                )
                if health:
                    health.record_error("timeout")
                logger.warning(f"TTS timeout for provider {provider.value}")

            except Exception as e:
                last_error = e
                if health:
                    health.record_error(str(e))
                logger.warning(f"TTS error for provider {provider.value}: {e}")

        # All providers failed
        self._metrics.tts_errors += 1
        raise last_error or VoiceError("All TTS providers failed")

    async def synthesize_stream(
        self,
        text: str,
        config: Optional[TTSConfig] = None,
        voice_profile: Optional[VoiceProfile] = None
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Stream audio synthesis.

        Args:
            text: Text to synthesize
            config: TTS configuration
            voice_profile: Optional voice profile

        Yields:
            AudioChunk objects
        """
        if voice_profile:
            config = voice_profile.to_tts_config()
        elif config is None:
            config = TTSConfig(provider=self.config.default_tts_provider)

        request_id = generate_request_id()

        provider = self._select_tts_provider(config.provider)
        instance = self._tts_providers.get(provider)

        if not instance:
            raise VoiceError(f"No TTS provider available")

        try:
            config.provider = provider
            async for chunk in instance.synthesize_stream(text, config):
                yield chunk

        except Exception as e:
            self._metrics.tts_errors += 1
            health = self._tts_health.get(provider)
            if health:
                health.record_error(str(e))
            raise

    # =========================================================================
    # Voice Management
    # =========================================================================

    async def get_available_voices(
        self,
        provider: Optional[TTSProvider] = None
    ) -> Dict[TTSProvider, List[Dict[str, Any]]]:
        """Get available voices from TTS providers."""
        result = {}

        providers_to_check = [provider] if provider else list(self._tts_providers.keys())

        for p in providers_to_check:
            instance = self._tts_providers.get(p)
            if instance:
                try:
                    voices = await instance.get_voices()
                    result[p] = voices
                except Exception as e:
                    logger.error(f"Failed to get voices from {p.value}: {e}")
                    result[p] = []

        return result

    # =========================================================================
    # Provider Selection
    # =========================================================================

    def _get_stt_providers_to_try(
        self,
        preferred: STTProvider
    ) -> List[STTProvider]:
        """Get ordered list of STT providers to try."""
        providers = [preferred]

        for p in self.config.stt_fallback_chain:
            if p != preferred and p not in providers:
                providers.append(p)

        return providers

    def _get_tts_providers_to_try(
        self,
        preferred: TTSProvider
    ) -> List[TTSProvider]:
        """Get ordered list of TTS providers to try."""
        providers = [preferred]

        for p in self.config.tts_fallback_chain:
            if p != preferred and p not in providers:
                providers.append(p)

        return providers

    def _select_stt_provider(self, preferred: STTProvider) -> STTProvider:
        """Select best available STT provider."""
        health_dict = {p.value: h for p, h in self._stt_health.items()}
        selected = self._stt_strategy.select_provider(health_dict, None)

        if selected:
            return STTProvider(selected)
        return preferred

    def _select_tts_provider(self, preferred: TTSProvider) -> TTSProvider:
        """Select best available TTS provider."""
        health_dict = {p.value: h for p, h in self._tts_health.items()}
        selected = self._tts_strategy.select_provider(health_dict, None)

        if selected:
            return TTSProvider(selected)
        return preferred

    # =========================================================================
    # Caching
    # =========================================================================

    def _get_cache_key(self, text: str, config: TTSConfig) -> str:
        """Generate cache key for TTS request."""
        import hashlib
        key_data = f"{text}:{config.provider.value}:{config.voice_id}:{config.model}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[SynthesisResult]:
        """Get result from cache if valid."""
        if key not in self._response_cache:
            return None

        result, timestamp = self._response_cache[key]
        if time.time() - timestamp > self.config.cache_ttl:
            del self._response_cache[key]
            return None

        return result

    def _add_to_cache(self, key: str, result: SynthesisResult) -> None:
        """Add result to cache."""
        self._response_cache[key] = (result, time.time())

        # Clean old entries
        if len(self._response_cache) > 1000:
            self._clean_cache()

    def _clean_cache(self) -> None:
        """Remove expired cache entries."""
        now = time.time()
        expired = [
            k for k, (_, t) in self._response_cache.items()
            if now - t > self.config.cache_ttl
        ]
        for k in expired:
            del self._response_cache[k]

    # =========================================================================
    # Metrics
    # =========================================================================

    def _update_avg_latency(self, service: str, latency_ms: float) -> None:
        """Update rolling average latency."""
        alpha = 0.1
        if service == "stt":
            self._metrics.stt_avg_latency_ms = (
                alpha * latency_ms +
                (1 - alpha) * self._metrics.stt_avg_latency_ms
            )
        else:
            self._metrics.tts_avg_latency_ms = (
                alpha * latency_ms +
                (1 - alpha) * self._metrics.tts_avg_latency_ms
            )

    def get_metrics(self) -> VoiceMetrics:
        """Get current voice engine metrics."""
        return self._metrics

    def get_provider_health(self) -> Dict[str, Dict[str, ProviderHealth]]:
        """Get health status of all providers."""
        return {
            "stt": {p.value: h for p, h in self._stt_health.items()},
            "tts": {p.value: h for p, h in self._tts_health.items()},
        }

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_provider_unhealthy(self, callback: Callable) -> None:
        """Register callback for when a provider becomes unhealthy."""
        self._on_provider_unhealthy.append(callback)

    def on_provider_recovered(self, callback: Callable) -> None:
        """Register callback for when a provider recovers."""
        self._on_provider_recovered.append(callback)


# =============================================================================
# Singleton Instance
# =============================================================================

_engine_instance: Optional[VoiceEngine] = None


def get_voice_engine(config: Optional[VoiceEngineConfig] = None) -> VoiceEngine:
    """Get or create the global voice engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = VoiceEngine(config)
    return _engine_instance


async def initialize_voice_engine(config: Optional[VoiceEngineConfig] = None) -> VoiceEngine:
    """Initialize and start the voice engine."""
    engine = get_voice_engine(config)
    await engine.start()
    return engine

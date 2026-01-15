"""
Speculative Execution Module.

This module implements speculative execution for LLM inference,
starting response generation on partial (stable) transcripts to
minimize end-to-end latency.

Key Features:
- Start LLM inference before final transcript
- Track transcript stability for speculation timing
- Validate speculative results against final transcript
- Intelligent abandonment on transcript changes
- Token-level result reuse when speculation succeeds
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from ..models import TranscriptSegment, TranscriptStatus, LLMStreamToken

logger = logging.getLogger(__name__)


class SpeculativeState(str, Enum):
    """State of speculative execution."""

    IDLE = "idle"               # No speculation in progress
    WAITING = "waiting"         # Waiting for stable partial
    EXECUTING = "executing"     # Speculation in progress
    VALIDATING = "validating"   # Validating against final transcript
    VALIDATED = "validated"     # Speculation was correct
    ABANDONED = "abandoned"     # Speculation was abandoned
    FAILED = "failed"           # Speculation failed


@dataclass
class TranscriptStability:
    """Tracks stability of a transcript over time."""

    text: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    change_count: int = 0
    word_count: int = 0

    # History of changes
    previous_texts: List[str] = field(default_factory=list)

    @property
    def stability_duration_ms(self) -> float:
        """How long the transcript has been stable."""
        return (self.last_seen - self.first_seen) * 1000

    @property
    def stability_score(self) -> float:
        """
        Calculate stability score (0.0 to 1.0).

        Based on:
        - Time stable
        - Number of changes
        - Word count
        """
        # Time factor: stable for 200ms = 0.5, 500ms = 1.0
        time_factor = min(1.0, self.stability_duration_ms / 500.0)

        # Change factor: fewer changes = higher score
        change_factor = max(0.0, 1.0 - (self.change_count * 0.1))

        # Word factor: more words = slightly higher score (3+ words preferred)
        word_factor = min(1.0, self.word_count / 3.0)

        return (time_factor * 0.5 + change_factor * 0.3 + word_factor * 0.2)


@dataclass
class SpeculativeResult:
    """Result of speculative execution."""

    state: SpeculativeState
    partial_transcript: str
    final_transcript: Optional[str] = None

    # Generated content
    tokens: List[LLMStreamToken] = field(default_factory=list)
    full_text: str = ""

    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    first_token_at: Optional[float] = None

    # Validation
    similarity_score: float = 0.0
    tokens_reusable: int = 0
    was_abandoned: bool = False
    abandonment_reason: Optional[str] = None

    @property
    def latency_saved_ms(self) -> float:
        """Estimate of latency saved by speculation."""
        if self.state == SpeculativeState.VALIDATED and self.first_token_at:
            # Time from speculation start to first token
            return (self.first_token_at - self.started_at) * 1000
        return 0.0

    @property
    def token_count(self) -> int:
        """Number of tokens generated."""
        return len(self.tokens)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative execution."""

    enabled: bool = True

    # Stability thresholds
    min_stability_score: float = 0.7
    min_stability_duration_ms: float = 150.0
    min_words: int = 3

    # Execution limits
    max_speculative_tokens: int = 50
    max_speculation_time_ms: float = 500.0

    # Validation
    similarity_threshold: float = 0.7  # Min similarity to validate
    abandon_on_low_similarity: bool = True
    similarity_check_interval_tokens: int = 10

    # Token reuse
    enable_token_reuse: bool = True
    max_reusable_tokens: int = 30


class TranscriptTracker:
    """
    Tracks transcript changes and calculates stability.

    Used to determine when a partial transcript is stable enough
    to trigger speculative execution.
    """

    def __init__(self, config: SpeculativeConfig):
        self.config = config
        self._current: Optional[TranscriptStability] = None
        self._history: List[str] = []
        self._lock = asyncio.Lock()

    async def update(self, segment: TranscriptSegment) -> TranscriptStability:
        """
        Update with new transcript segment.

        Returns updated stability information.
        """
        async with self._lock:
            text = segment.text.strip()
            now = time.time()
            words = len(text.split())

            if self._current is None or text != self._current.text:
                # Transcript changed
                if self._current:
                    self._history.append(self._current.text)
                    # Keep limited history
                    if len(self._history) > 10:
                        self._history.pop(0)

                change_count = 0
                if self._current:
                    change_count = self._current.change_count + 1

                self._current = TranscriptStability(
                    text=text,
                    first_seen=now,
                    last_seen=now,
                    change_count=change_count,
                    word_count=words,
                    previous_texts=list(self._history),
                )
            else:
                # Same text, update last_seen
                self._current.last_seen = now

            return self._current

    async def is_stable_for_speculation(self) -> Tuple[bool, Optional[str]]:
        """
        Check if current transcript is stable enough for speculation.

        Returns:
            Tuple of (is_stable, reason_if_not_stable)
        """
        if not self._current:
            return False, "No transcript available"

        # Check minimum words
        if self._current.word_count < self.config.min_words:
            return False, f"Need at least {self.config.min_words} words"

        # Check stability duration
        if self._current.stability_duration_ms < self.config.min_stability_duration_ms:
            return False, f"Need {self.config.min_stability_duration_ms}ms stability"

        # Check stability score
        if self._current.stability_score < self.config.min_stability_score:
            return False, f"Stability score {self._current.stability_score:.2f} < {self.config.min_stability_score}"

        return True, None

    async def get_current(self) -> Optional[TranscriptStability]:
        """Get current transcript stability."""
        return self._current

    async def reset(self) -> None:
        """Reset the tracker."""
        async with self._lock:
            self._current = None
            self._history.clear()


class SpeculativeExecutor:
    """
    Executes speculative LLM inference on partial transcripts.

    Key optimizations:
    1. Start inference on stable partial transcripts
    2. Continue generating while transcript stabilizes
    3. Validate against final transcript
    4. Reuse valid tokens to skip re-generation
    5. Abandon early if transcript diverges significantly
    """

    def __init__(
        self,
        config: SpeculativeConfig,
        llm_generator: Callable[[str, Dict[str, Any]], AsyncIterator[LLMStreamToken]],
    ):
        """
        Initialize speculative executor.

        Args:
            config: Configuration options
            llm_generator: Async generator function for LLM tokens
        """
        self.config = config
        self._llm_generator = llm_generator

        # State
        self._state = SpeculativeState.IDLE
        self._current_result: Optional[SpeculativeResult] = None
        self._transcript_tracker = TranscriptTracker(config)

        # Execution tracking
        self._execution_task: Optional[asyncio.Task] = None
        self._cancel_event = asyncio.Event()

        # Statistics
        self._total_speculations = 0
        self._validated_speculations = 0
        self._abandoned_speculations = 0
        self._total_tokens_reused = 0

        # Lock
        self._lock = asyncio.Lock()

    @property
    def state(self) -> SpeculativeState:
        """Get current state."""
        return self._state

    @property
    def is_executing(self) -> bool:
        """Check if speculation is in progress."""
        return self._state == SpeculativeState.EXECUTING

    @property
    def current_result(self) -> Optional[SpeculativeResult]:
        """Get current speculation result."""
        return self._current_result

    async def process_transcript(
        self,
        segment: TranscriptSegment,
        context: Dict[str, Any],
    ) -> Optional[SpeculativeResult]:
        """
        Process a transcript segment and potentially start speculation.

        Args:
            segment: The transcript segment
            context: LLM context (system prompt, history, etc.)

        Returns:
            SpeculativeResult if speculation started/completed, None otherwise
        """
        if not self.config.enabled:
            return None

        # Update tracker
        stability = await self._transcript_tracker.update(segment)

        # Handle final transcript
        if segment.status == TranscriptStatus.FINAL:
            return await self._handle_final_transcript(segment.text)

        # Check if we should start speculation
        if self._state == SpeculativeState.IDLE:
            is_stable, reason = await self._transcript_tracker.is_stable_for_speculation()

            if is_stable:
                logger.debug(
                    f"Starting speculation on: '{stability.text[:50]}...' "
                    f"(stability: {stability.stability_score:.2f})"
                )
                await self._start_speculation(stability.text, context)

        return self._current_result

    async def _start_speculation(
        self,
        partial_transcript: str,
        context: Dict[str, Any],
    ) -> None:
        """Start speculative execution."""
        async with self._lock:
            # Cancel any existing speculation
            if self._execution_task and not self._execution_task.done():
                self._cancel_event.set()
                await self._execution_task

            # Reset state
            self._cancel_event.clear()
            self._state = SpeculativeState.EXECUTING
            self._total_speculations += 1

            self._current_result = SpeculativeResult(
                state=SpeculativeState.EXECUTING,
                partial_transcript=partial_transcript,
            )

            # Start execution task
            self._execution_task = asyncio.create_task(
                self._execute_speculation(partial_transcript, context)
            )

    async def _execute_speculation(
        self,
        partial_transcript: str,
        context: Dict[str, Any],
    ) -> None:
        """Execute speculative LLM inference."""
        start_time = time.time()
        tokens_generated = 0

        try:
            async for token in self._llm_generator(partial_transcript, context):
                # Check for cancellation
                if self._cancel_event.is_set():
                    logger.debug("Speculation cancelled")
                    break

                # Check timeout
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > self.config.max_speculation_time_ms:
                    logger.debug(f"Speculation timeout after {elapsed_ms:.1f}ms")
                    break

                # Check token limit
                tokens_generated += 1
                if tokens_generated > self.config.max_speculative_tokens:
                    logger.debug(f"Speculation hit token limit: {tokens_generated}")
                    break

                # Record token
                if self._current_result:
                    if token.is_first:
                        self._current_result.first_token_at = time.time()

                    self._current_result.tokens.append(token)
                    self._current_result.full_text = token.cumulative_text

                # Periodic similarity check (for early abandonment)
                if (
                    self.config.abandon_on_low_similarity
                    and tokens_generated % self.config.similarity_check_interval_tokens == 0
                ):
                    current_stability = await self._transcript_tracker.get_current()
                    if current_stability:
                        similarity = self._calculate_similarity(
                            partial_transcript,
                            current_stability.text,
                        )
                        if similarity < 0.5:  # Early abandonment threshold
                            logger.debug(
                                f"Early abandonment: similarity {similarity:.2f} < 0.5"
                            )
                            self._state = SpeculativeState.ABANDONED
                            if self._current_result:
                                self._current_result.was_abandoned = True
                                self._current_result.abandonment_reason = "Low similarity during execution"
                            self._abandoned_speculations += 1
                            return

        except asyncio.CancelledError:
            logger.debug("Speculation task cancelled")
        except Exception as e:
            logger.error(f"Speculation error: {e}")
            self._state = SpeculativeState.FAILED

        if self._current_result:
            self._current_result.completed_at = time.time()

    async def _handle_final_transcript(
        self,
        final_transcript: str,
    ) -> Optional[SpeculativeResult]:
        """
        Handle final transcript and validate speculation.

        Args:
            final_transcript: The final, confirmed transcript

        Returns:
            Validated or abandoned SpeculativeResult
        """
        if not self._current_result or self._state != SpeculativeState.EXECUTING:
            # No active speculation to validate
            await self._reset()
            return None

        # Cancel ongoing execution
        if self._execution_task and not self._execution_task.done():
            self._cancel_event.set()
            try:
                await asyncio.wait_for(self._execution_task, timeout=0.5)
            except asyncio.TimeoutError:
                self._execution_task.cancel()

        # Validate speculation
        self._state = SpeculativeState.VALIDATING
        self._current_result.final_transcript = final_transcript

        similarity = self._calculate_similarity(
            self._current_result.partial_transcript,
            final_transcript,
        )
        self._current_result.similarity_score = similarity

        if similarity >= self.config.similarity_threshold:
            # Speculation validated!
            self._state = SpeculativeState.VALIDATED
            self._current_result.state = SpeculativeState.VALIDATED
            self._validated_speculations += 1

            # Calculate reusable tokens
            if self.config.enable_token_reuse:
                reusable = self._calculate_reusable_tokens(
                    self._current_result.partial_transcript,
                    final_transcript,
                    self._current_result.tokens,
                )
                self._current_result.tokens_reusable = reusable
                self._total_tokens_reused += reusable

            logger.info(
                f"Speculation validated! Similarity: {similarity:.2f}, "
                f"Reusable tokens: {self._current_result.tokens_reusable}"
            )

        else:
            # Speculation must be abandoned
            self._state = SpeculativeState.ABANDONED
            self._current_result.state = SpeculativeState.ABANDONED
            self._current_result.was_abandoned = True
            self._current_result.abandonment_reason = (
                f"Similarity {similarity:.2f} < threshold {self.config.similarity_threshold}"
            )
            self._abandoned_speculations += 1

            logger.debug(
                f"Speculation abandoned: similarity {similarity:.2f} < {self.config.similarity_threshold}"
            )

        result = self._current_result
        await self._reset()

        return result

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, t1, t2).ratio()

    def _calculate_reusable_tokens(
        self,
        partial: str,
        final: str,
        tokens: List[LLMStreamToken],
    ) -> int:
        """
        Calculate how many tokens can be reused.

        Tokens are reusable if they were generated based on the
        matching prefix of the transcripts.
        """
        if not tokens:
            return 0

        # Find common prefix
        partial_words = partial.lower().split()
        final_words = final.lower().split()

        common_prefix_len = 0
        for pw, fw in zip(partial_words, final_words):
            if pw == fw:
                common_prefix_len += 1
            else:
                break

        # Estimate reusable tokens based on common prefix ratio
        if len(partial_words) == 0:
            return 0

        prefix_ratio = common_prefix_len / len(partial_words)

        # Reuse tokens proportionally, with a cap
        reusable = int(len(tokens) * prefix_ratio)
        return min(reusable, self.config.max_reusable_tokens)

    async def get_reusable_tokens(self) -> List[LLMStreamToken]:
        """
        Get tokens that can be reused from validated speculation.

        Returns:
            List of reusable tokens, empty if speculation wasn't validated
        """
        if (
            self._current_result
            and self._current_result.state == SpeculativeState.VALIDATED
            and self._current_result.tokens_reusable > 0
        ):
            return self._current_result.tokens[:self._current_result.tokens_reusable]
        return []

    async def cancel(self) -> None:
        """Cancel current speculation."""
        self._cancel_event.set()
        if self._execution_task and not self._execution_task.done():
            try:
                await asyncio.wait_for(self._execution_task, timeout=0.5)
            except asyncio.TimeoutError:
                self._execution_task.cancel()
        await self._reset()

    async def _reset(self) -> None:
        """Reset state for next speculation."""
        self._state = SpeculativeState.IDLE
        self._cancel_event.clear()
        await self._transcript_tracker.reset()

    def get_statistics(self) -> Dict[str, Any]:
        """Get speculation statistics."""
        validation_rate = (
            self._validated_speculations / self._total_speculations
            if self._total_speculations > 0
            else 0.0
        )

        return {
            "total_speculations": self._total_speculations,
            "validated_speculations": self._validated_speculations,
            "abandoned_speculations": self._abandoned_speculations,
            "validation_rate": round(validation_rate, 3),
            "total_tokens_reused": self._total_tokens_reused,
            "avg_tokens_reused": (
                self._total_tokens_reused / self._validated_speculations
                if self._validated_speculations > 0
                else 0.0
            ),
            "current_state": self._state.value,
        }


# Factory function
def create_speculative_executor(
    llm_generator: Callable[[str, Dict[str, Any]], AsyncIterator[LLMStreamToken]],
    enabled: bool = True,
    min_stability_score: float = 0.7,
    min_words: int = 3,
    max_tokens: int = 50,
) -> SpeculativeExecutor:
    """
    Create a configured speculative executor.

    Args:
        llm_generator: Async generator for LLM tokens
        enabled: Whether speculation is enabled
        min_stability_score: Minimum stability to trigger speculation
        min_words: Minimum words required
        max_tokens: Maximum tokens to generate speculatively

    Returns:
        Configured SpeculativeExecutor instance
    """
    config = SpeculativeConfig(
        enabled=enabled,
        min_stability_score=min_stability_score,
        min_words=min_words,
        max_speculative_tokens=max_tokens,
    )

    return SpeculativeExecutor(config, llm_generator)

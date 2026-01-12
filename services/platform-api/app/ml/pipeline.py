"""
ML Pipeline Framework

Orchestrates ML models for conversation processing:
- Modular pipeline stages
- Batch processing
- Model caching
- Performance optimization
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import time
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class PipelineStageStatus(str, Enum):
    """Pipeline stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage_name: str
    status: PipelineStageStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline."""
    name: str
    max_concurrent: int = 10
    timeout_seconds: float = 30.0
    retry_failed: bool = True
    max_retries: int = 3
    cache_results: bool = True
    cache_ttl_seconds: int = 3600
    enable_metrics: bool = True


@dataclass
class PipelineInput:
    """Input to the ML pipeline."""
    text: str
    conversation_id: Optional[str] = None
    turn_id: Optional[str] = None
    speaker: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Optional[List[str]] = None  # Previous utterances for context


@dataclass
class PipelineOutput:
    """Output from the ML pipeline."""
    input_id: str
    stages: List[StageResult]
    total_duration_ms: float
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)

    def get_result(self, stage_name: str) -> Optional[Any]:
        """Get result from a specific stage."""
        return self.results.get(stage_name)


class PipelineStage(ABC, Generic[T, R]):
    """Abstract base for pipeline stages."""

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        timeout: float = 10.0,
        depends_on: Optional[List[str]] = None,
    ):
        self.name = name
        self.enabled = enabled
        self.timeout = timeout
        self.depends_on = depends_on or []

    @abstractmethod
    async def process(self, input_data: T, context: Dict[str, Any]) -> R:
        """Process the input data."""
        pass

    async def execute(self, input_data: T, context: Dict[str, Any]) -> StageResult:
        """Execute the stage with timing and error handling."""
        if not self.enabled:
            return StageResult(
                stage_name=self.name,
                status=PipelineStageStatus.SKIPPED,
            )

        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                self.process(input_data, context),
                timeout=self.timeout,
            )

            return StageResult(
                stage_name=self.name,
                status=PipelineStageStatus.COMPLETED,
                output=result,
                duration_ms=(time.time() - start_time) * 1000,
            )

        except asyncio.TimeoutError:
            return StageResult(
                stage_name=self.name,
                status=PipelineStageStatus.FAILED,
                error=f"Stage timed out after {self.timeout}s",
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Stage {self.name} failed: {e}")
            return StageResult(
                stage_name=self.name,
                status=PipelineStageStatus.FAILED,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )


class MLPipeline:
    """
    ML processing pipeline for conversations.

    Orchestrates multiple ML stages for comprehensive analysis.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig(name="default")
        self._stages: Dict[str, PipelineStage] = {}
        self._stage_order: List[str] = []
        self._cache: Dict[str, PipelineOutput] = {}
        self._metrics = PipelineMetrics()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    def add_stage(self, stage: PipelineStage) -> "MLPipeline":
        """Add a stage to the pipeline."""
        self._stages[stage.name] = stage
        if stage.name not in self._stage_order:
            self._stage_order.append(stage.name)
        return self

    def remove_stage(self, name: str) -> "MLPipeline":
        """Remove a stage from the pipeline."""
        if name in self._stages:
            del self._stages[name]
            self._stage_order.remove(name)
        return self

    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get a stage by name."""
        return self._stages.get(name)

    def _get_execution_order(self) -> List[str]:
        """Get stages in dependency-aware order."""
        # Topological sort based on dependencies
        visited = set()
        order = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)

            stage = self._stages.get(name)
            if stage:
                for dep in stage.depends_on:
                    if dep in self._stages:
                        visit(dep)
                order.append(name)

        for name in self._stage_order:
            visit(name)

        return order

    async def process(self, input_data: PipelineInput) -> PipelineOutput:
        """Process input through all pipeline stages."""
        async with self._semaphore:
            return await self._execute_pipeline(input_data)

    async def _execute_pipeline(self, input_data: PipelineInput) -> PipelineOutput:
        """Execute the full pipeline."""
        start_time = time.time()
        input_id = f"{input_data.conversation_id or 'none'}:{input_data.turn_id or str(time.time())}"

        # Check cache
        if self.config.cache_results:
            cache_key = self._make_cache_key(input_data)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for {cache_key}")
                return self._cache[cache_key]

        # Initialize context with input
        context: Dict[str, Any] = {
            "input": input_data,
            "text": input_data.text,
            "metadata": input_data.metadata,
            "conversation_context": input_data.context or [],
        }

        # Execute stages in order
        stage_results = []
        results = {}
        all_success = True

        for stage_name in self._get_execution_order():
            stage = self._stages[stage_name]

            # Check dependencies
            deps_satisfied = all(
                results.get(dep) is not None
                for dep in stage.depends_on
            )

            if not deps_satisfied:
                stage_results.append(StageResult(
                    stage_name=stage_name,
                    status=PipelineStageStatus.SKIPPED,
                    error="Dependencies not satisfied",
                ))
                continue

            # Execute stage
            result = await stage.execute(input_data.text, context)
            stage_results.append(result)

            if result.status == PipelineStageStatus.COMPLETED:
                results[stage_name] = result.output
                context[stage_name] = result.output
            elif result.status == PipelineStageStatus.FAILED:
                all_success = False

                # Retry if configured
                if self.config.retry_failed and self.config.max_retries > 0:
                    for attempt in range(self.config.max_retries):
                        logger.info(f"Retrying {stage_name}, attempt {attempt + 1}")
                        result = await stage.execute(input_data.text, context)
                        if result.status == PipelineStageStatus.COMPLETED:
                            results[stage_name] = result.output
                            context[stage_name] = result.output
                            all_success = True
                            break
                        await asyncio.sleep(0.1 * (attempt + 1))

        # Create output
        output = PipelineOutput(
            input_id=input_id,
            stages=stage_results,
            total_duration_ms=(time.time() - start_time) * 1000,
            success=all_success,
            results=results,
        )

        # Cache result
        if self.config.cache_results:
            cache_key = self._make_cache_key(input_data)
            self._cache[cache_key] = output

        # Record metrics
        if self.config.enable_metrics:
            self._metrics.record_execution(output)

        return output

    def _make_cache_key(self, input_data: PipelineInput) -> str:
        """Create cache key from input."""
        import hashlib
        content = f"{input_data.text}:{input_data.conversation_id}:{input_data.turn_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def process_batch(
        self,
        inputs: List[PipelineInput],
        max_concurrent: Optional[int] = None,
    ) -> List[PipelineOutput]:
        """Process multiple inputs concurrently."""
        max_concurrent = max_concurrent or self.config.max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(input_data: PipelineInput) -> PipelineOutput:
            async with semaphore:
                return await self._execute_pipeline(input_data)

        return await asyncio.gather(*[process_one(inp) for inp in inputs])

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return self._metrics.to_dict()

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()


@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_duration_ms: float = 0.0
    stage_durations: Dict[str, List[float]] = field(default_factory=lambda: {})

    def record_execution(self, output: PipelineOutput) -> None:
        """Record an execution."""
        self.total_executions += 1
        self.total_duration_ms += output.total_duration_ms

        if output.success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        for stage_result in output.stages:
            if stage_result.stage_name not in self.stage_durations:
                self.stage_durations[stage_result.stage_name] = []
            self.stage_durations[stage_result.stage_name].append(stage_result.duration_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        import statistics

        stage_stats = {}
        for stage, durations in self.stage_durations.items():
            if durations:
                stage_stats[stage] = {
                    "count": len(durations),
                    "avg_ms": statistics.mean(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                    "p95_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) >= 20 else max(durations),
                }

        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.successful_executions / self.total_executions if self.total_executions > 0 else 0,
            "avg_duration_ms": self.total_duration_ms / self.total_executions if self.total_executions > 0 else 0,
            "stage_stats": stage_stats,
        }


class BatchProcessor:
    """
    Batch processor for large-scale ML processing.

    Features:
    - Chunked processing
    - Progress tracking
    - Error handling
    - Result aggregation
    """

    def __init__(
        self,
        pipeline: MLPipeline,
        batch_size: int = 100,
        max_workers: int = 4,
    ):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._progress: Dict[str, Any] = {}

    async def process_texts(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[PipelineOutput]:
        """Process a list of texts."""
        inputs = [
            PipelineInput(text=text, turn_id=str(i))
            for i, text in enumerate(texts)
        ]
        return await self.process_inputs(inputs, progress_callback)

    async def process_inputs(
        self,
        inputs: List[PipelineInput],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[PipelineOutput]:
        """Process a list of inputs in batches."""
        results = []
        total = len(inputs)

        for i in range(0, total, self.batch_size):
            batch = inputs[i:i + self.batch_size]
            batch_results = await self.pipeline.process_batch(
                batch,
                max_concurrent=self.max_workers,
            )
            results.extend(batch_results)

            if progress_callback:
                progress_callback(min(i + self.batch_size, total), total)

        return results

    async def process_conversation(
        self,
        messages: List[Dict[str, str]],
        conversation_id: Optional[str] = None,
    ) -> List[PipelineOutput]:
        """Process a full conversation with context."""
        inputs = []
        context_window = []

        for i, msg in enumerate(messages):
            text = msg.get("content", msg.get("text", ""))
            speaker = msg.get("role", msg.get("speaker"))

            input_data = PipelineInput(
                text=text,
                conversation_id=conversation_id,
                turn_id=str(i),
                speaker=speaker,
                context=list(context_window),
            )
            inputs.append(input_data)

            # Add to context window (keep last 5)
            context_window.append(text)
            if len(context_window) > 5:
                context_window.pop(0)

        return await self.process_inputs(inputs)


class ConversationAnalyzer:
    """
    High-level conversation analyzer.

    Combines multiple ML capabilities for comprehensive analysis.
    """

    def __init__(self, pipeline: MLPipeline):
        self.pipeline = pipeline

    async def analyze(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze a full conversation."""
        processor = BatchProcessor(self.pipeline)
        results = await processor.process_conversation(conversation)

        # Aggregate results
        analysis = {
            "turn_count": len(results),
            "successful_analyses": sum(1 for r in results if r.success),
            "turns": [],
            "overall": {},
        }

        # Collect per-turn results
        all_intents = []
        all_sentiments = []
        all_entities = []

        for i, result in enumerate(results):
            turn_analysis = {
                "turn_id": i,
                "text": conversation[i].get("content", ""),
                "speaker": conversation[i].get("role", ""),
            }

            # Add stage results
            for stage_name, stage_output in result.results.items():
                turn_analysis[stage_name] = stage_output

                # Collect for aggregation
                if stage_name == "intent" and stage_output:
                    all_intents.append(stage_output)
                if stage_name == "sentiment" and stage_output:
                    all_sentiments.append(stage_output)
                if stage_name == "entities" and stage_output:
                    all_entities.extend(stage_output)

            analysis["turns"].append(turn_analysis)

        # Overall statistics
        analysis["overall"] = {
            "intents": self._aggregate_intents(all_intents),
            "sentiment_trend": self._calculate_sentiment_trend(all_sentiments),
            "entities": self._aggregate_entities(all_entities),
        }

        return analysis

    def _aggregate_intents(self, intents: List[Any]) -> Dict[str, int]:
        """Aggregate intents across turns."""
        from collections import Counter
        intent_counts = Counter()
        for intent in intents:
            if hasattr(intent, 'label'):
                intent_counts[intent.label] += 1
            elif isinstance(intent, dict):
                intent_counts[intent.get('label', 'unknown')] += 1
        return dict(intent_counts)

    def _calculate_sentiment_trend(self, sentiments: List[Any]) -> List[float]:
        """Calculate sentiment trend."""
        scores = []
        for sentiment in sentiments:
            if hasattr(sentiment, 'score'):
                scores.append(sentiment.score)
            elif isinstance(sentiment, dict):
                scores.append(sentiment.get('score', 0))
        return scores

    def _aggregate_entities(self, entities: List[Any]) -> Dict[str, List[str]]:
        """Aggregate entities by type."""
        from collections import defaultdict
        by_type = defaultdict(set)
        for entity in entities:
            if hasattr(entity, 'type') and hasattr(entity, 'text'):
                by_type[entity.type].add(entity.text)
            elif isinstance(entity, dict):
                by_type[entity.get('type', 'unknown')].add(entity.get('text', ''))
        return {k: list(v) for k, v in by_type.items()}

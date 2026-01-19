"""A/B testing experiment manager."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import redis.asyncio as redis

from .models import (
    Experiment,
    ExperimentVariant,
    ExperimentStatus,
    ExperimentMetric,
    MetricType,
    VariantAssignment,
)

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manages A/B testing experiments for AI voice agents.

    Features:
    - Experiment CRUD
    - Variant management
    - Assignment tracking
    - Metric recording
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
    ):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None

        # In-memory cache
        self._experiments: Dict[str, Experiment] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the experiment manager."""
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        await self._load_experiments()
        logger.info("Experiment manager started")

    async def stop(self) -> None:
        """Stop the experiment manager."""
        if self.redis:
            await self.redis.close()
        logger.info("Experiment manager stopped")

    # Experiment CRUD

    async def create_experiment(
        self,
        organization_id: str,
        name: str,
        variants: List[Dict[str, Any]],
        metrics: List[Dict[str, Any]],
        description: str = "",
        hypothesis: str = "",
        phone_number_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        traffic_percentage: float = 100.0,
        min_sample_size: int = 100,
        confidence_level: float = 0.95,
        created_by: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Create a new experiment."""
        experiment_id = f"exp_{uuid4().hex[:12]}"

        # Create variants
        experiment_variants = []
        for i, v in enumerate(variants):
            variant = ExperimentVariant(
                id=f"var_{uuid4().hex[:8]}",
                name=v.get("name", f"Variant {i + 1}"),
                description=v.get("description", ""),
                agent_id=v.get("agent_id", ""),
                traffic_percentage=v.get("traffic_percentage", 50.0),
                is_control=v.get("is_control", i == 0),
                agent_config_overrides=v.get("agent_config_overrides", {}),
            )
            experiment_variants.append(variant)

        # Create metrics
        experiment_metrics = []
        for i, m in enumerate(metrics):
            metric = ExperimentMetric(
                metric_type=MetricType(m.get("metric_type", "success_rate")),
                name=m.get("name", ""),
                description=m.get("description", ""),
                is_primary=m.get("is_primary", i == 0),
                target_value=m.get("target_value"),
                minimum_improvement=m.get("minimum_improvement", 0.0),
            )
            experiment_metrics.append(metric)

        experiment = Experiment(
            id=experiment_id,
            organization_id=organization_id,
            name=name,
            description=description,
            hypothesis=hypothesis,
            variants=experiment_variants,
            metrics=experiment_metrics,
            phone_number_ids=phone_number_ids or [],
            agent_ids=agent_ids or [],
            traffic_percentage=traffic_percentage,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            created_by=created_by,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Validate
        errors = experiment.validate()
        if errors:
            raise ValueError(f"Invalid experiment: {'; '.join(errors)}")

        # Store
        async with self._lock:
            self._experiments[experiment_id] = experiment

        await self._persist_experiment(experiment)

        logger.info(f"Created experiment: {experiment_id} ({name})")
        return experiment

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        async with self._lock:
            return self._experiments.get(experiment_id)

    async def list_experiments(
        self,
        organization_id: str,
        status: Optional[ExperimentStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Experiment], int]:
        """List experiments for an organization."""
        async with self._lock:
            experiments = [
                e for e in self._experiments.values()
                if e.organization_id == organization_id
            ]

        if status:
            experiments = [e for e in experiments if e.status == status]

        experiments.sort(key=lambda e: e.created_at, reverse=True)
        total = len(experiments)
        experiments = experiments[offset:offset + limit]

        return experiments, total

    async def update_experiment(
        self,
        experiment_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Experiment]:
        """Update an experiment."""
        async with self._lock:
            if experiment_id not in self._experiments:
                return None

            experiment = self._experiments[experiment_id]

            # Only allow updates for draft experiments
            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError("Can only update draft experiments")

            for key, value in updates.items():
                if hasattr(experiment, key):
                    setattr(experiment, key, value)

            experiment.updated_at = datetime.utcnow()
            self._experiments[experiment_id] = experiment

        await self._persist_experiment(experiment)
        return experiment

    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        async with self._lock:
            if experiment_id not in self._experiments:
                return False

            experiment = self._experiments[experiment_id]

            if experiment.status == ExperimentStatus.RUNNING:
                raise ValueError("Cannot delete running experiment")

            del self._experiments[experiment_id]

        if self.redis:
            await self.redis.hdel(f"experiments:{experiment.organization_id}", experiment_id)

        logger.info(f"Deleted experiment: {experiment_id}")
        return True

    # Experiment lifecycle

    async def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        async with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError("Experiment not found")

            experiment = self._experiments[experiment_id]

            if experiment.status not in (ExperimentStatus.DRAFT, ExperimentStatus.PAUSED):
                raise ValueError(f"Cannot start experiment in {experiment.status.value} status")

            errors = experiment.validate()
            if errors:
                raise ValueError(f"Invalid experiment: {'; '.join(errors)}")

            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = datetime.utcnow()
            experiment.updated_at = datetime.utcnow()

            self._experiments[experiment_id] = experiment

        await self._persist_experiment(experiment)
        logger.info(f"Started experiment: {experiment_id}")
        return experiment

    async def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause a running experiment."""
        async with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError("Experiment not found")

            experiment = self._experiments[experiment_id]

            if experiment.status != ExperimentStatus.RUNNING:
                raise ValueError("Experiment is not running")

            experiment.status = ExperimentStatus.PAUSED
            experiment.updated_at = datetime.utcnow()

            self._experiments[experiment_id] = experiment

        await self._persist_experiment(experiment)
        logger.info(f"Paused experiment: {experiment_id}")
        return experiment

    async def complete_experiment(
        self,
        experiment_id: str,
        winning_variant_id: Optional[str] = None,
    ) -> Experiment:
        """Complete an experiment."""
        async with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError("Experiment not found")

            experiment = self._experiments[experiment_id]

            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.utcnow()
            experiment.updated_at = datetime.utcnow()

            self._experiments[experiment_id] = experiment

        await self._persist_experiment(experiment)
        logger.info(f"Completed experiment: {experiment_id}")
        return experiment

    async def cancel_experiment(self, experiment_id: str) -> Experiment:
        """Cancel an experiment."""
        async with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError("Experiment not found")

            experiment = self._experiments[experiment_id]

            experiment.status = ExperimentStatus.CANCELED
            experiment.updated_at = datetime.utcnow()

            self._experiments[experiment_id] = experiment

        await self._persist_experiment(experiment)
        logger.info(f"Canceled experiment: {experiment_id}")
        return experiment

    # Variant management

    async def add_variant(
        self,
        experiment_id: str,
        variant_data: Dict[str, Any],
    ) -> ExperimentVariant:
        """Add a variant to an experiment."""
        async with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError("Experiment not found")

            experiment = self._experiments[experiment_id]

            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError("Can only add variants to draft experiments")

            variant = ExperimentVariant(
                id=f"var_{uuid4().hex[:8]}",
                name=variant_data.get("name", f"Variant {len(experiment.variants) + 1}"),
                description=variant_data.get("description", ""),
                agent_id=variant_data.get("agent_id", ""),
                traffic_percentage=variant_data.get("traffic_percentage", 0.0),
                is_control=variant_data.get("is_control", False),
                agent_config_overrides=variant_data.get("agent_config_overrides", {}),
            )

            experiment.variants.append(variant)
            experiment.updated_at = datetime.utcnow()

            self._experiments[experiment_id] = experiment

        await self._persist_experiment(experiment)
        return variant

    async def update_variant(
        self,
        experiment_id: str,
        variant_id: str,
        updates: Dict[str, Any],
    ) -> Optional[ExperimentVariant]:
        """Update a variant."""
        async with self._lock:
            if experiment_id not in self._experiments:
                return None

            experiment = self._experiments[experiment_id]

            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError("Can only update variants in draft experiments")

            for variant in experiment.variants:
                if variant.id == variant_id:
                    for key, value in updates.items():
                        if hasattr(variant, key):
                            setattr(variant, key, value)

                    experiment.updated_at = datetime.utcnow()
                    self._experiments[experiment_id] = experiment
                    await self._persist_experiment(experiment)
                    return variant

        return None

    async def remove_variant(
        self,
        experiment_id: str,
        variant_id: str,
    ) -> bool:
        """Remove a variant from an experiment."""
        async with self._lock:
            if experiment_id not in self._experiments:
                return False

            experiment = self._experiments[experiment_id]

            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError("Can only remove variants from draft experiments")

            original_count = len(experiment.variants)
            experiment.variants = [v for v in experiment.variants if v.id != variant_id]

            if len(experiment.variants) == original_count:
                return False

            experiment.updated_at = datetime.utcnow()
            self._experiments[experiment_id] = experiment

        await self._persist_experiment(experiment)
        return True

    # Assignment tracking

    async def record_assignment(
        self,
        experiment_id: str,
        variant_id: str,
        call_id: str,
        phone_number: str = "",
        contact_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VariantAssignment:
        """Record a call assignment to a variant."""
        assignment = VariantAssignment(
            id=f"asn_{uuid4().hex[:12]}",
            experiment_id=experiment_id,
            variant_id=variant_id,
            call_id=call_id,
            contact_id=contact_id,
            phone_number=phone_number,
            metadata=metadata or {},
        )

        # Store assignment
        if self.redis:
            await self.redis.hset(
                f"experiment_assignments:{experiment_id}",
                call_id,
                json.dumps(assignment.to_dict()),
            )

            # Increment participant count
            await self.redis.hincrby(f"experiment_variants:{experiment_id}", variant_id, 1)

        # Update in-memory counts
        async with self._lock:
            if experiment_id in self._experiments:
                experiment = self._experiments[experiment_id]
                experiment.total_participants += 1

                for variant in experiment.variants:
                    if variant.id == variant_id:
                        variant.total_calls += 1
                        break

        return assignment

    async def get_assignment(
        self,
        experiment_id: str,
        call_id: str,
    ) -> Optional[VariantAssignment]:
        """Get assignment for a call."""
        if not self.redis:
            return None

        data = await self.redis.hget(f"experiment_assignments:{experiment_id}", call_id)
        if not data:
            return None

        try:
            assignment_dict = json.loads(data)
            assignment_dict["assigned_at"] = datetime.fromisoformat(assignment_dict["assigned_at"])
            return VariantAssignment(**assignment_dict)
        except Exception:
            return None

    # Metric recording

    async def record_call_result(
        self,
        experiment_id: str,
        variant_id: str,
        call_id: str,
        success: bool,
        duration: int,
        sentiment_score: float = 0.0,
        converted: bool = False,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record call result for experiment metrics."""
        async with self._lock:
            if experiment_id not in self._experiments:
                return

            experiment = self._experiments[experiment_id]

            for variant in experiment.variants:
                if variant.id == variant_id:
                    if success:
                        variant.successful_calls += 1
                    variant.total_duration += duration
                    variant.total_sentiment += sentiment_score

                    if converted:
                        variant.conversions += 1

                    if custom_metrics:
                        for key, value in custom_metrics.items():
                            variant.custom_metrics[key] = (
                                variant.custom_metrics.get(key, 0.0) + value
                            )
                    break

            self._experiments[experiment_id] = experiment

        await self._persist_experiment(experiment)

        # Store in Redis for persistence
        if self.redis:
            result_data = {
                "success": success,
                "duration": duration,
                "sentiment_score": sentiment_score,
                "converted": converted,
                "custom_metrics": custom_metrics or {},
                "recorded_at": datetime.utcnow().isoformat(),
            }
            await self.redis.hset(
                f"experiment_results:{experiment_id}:{variant_id}",
                call_id,
                json.dumps(result_data),
            )

    async def get_active_experiments_for_phone(
        self,
        organization_id: str,
        phone_number_id: str,
    ) -> List[Experiment]:
        """Get active experiments that include a phone number."""
        async with self._lock:
            experiments = [
                e for e in self._experiments.values()
                if (
                    e.organization_id == organization_id
                    and e.status == ExperimentStatus.RUNNING
                    and (
                        not e.phone_number_ids
                        or phone_number_id in e.phone_number_ids
                    )
                )
            ]

        return experiments

    async def get_active_experiments_for_agent(
        self,
        organization_id: str,
        agent_id: str,
    ) -> List[Experiment]:
        """Get active experiments that include an agent."""
        async with self._lock:
            experiments = [
                e for e in self._experiments.values()
                if (
                    e.organization_id == organization_id
                    and e.status == ExperimentStatus.RUNNING
                    and (not e.agent_ids or agent_id in e.agent_ids)
                )
            ]

        return experiments

    # Persistence

    async def _persist_experiment(self, experiment: Experiment) -> None:
        """Persist experiment to Redis."""
        if not self.redis:
            return

        await self.redis.hset(
            f"experiments:{experiment.organization_id}",
            experiment.id,
            json.dumps(experiment.to_dict()),
        )

    async def _load_experiments(self) -> None:
        """Load experiments from Redis."""
        if not self.redis:
            return

        async for key in self.redis.scan_iter(match="experiments:*"):
            experiment_data = await self.redis.hgetall(key)

            for exp_id, data in experiment_data.items():
                try:
                    exp_dict = json.loads(data)
                    exp_dict["status"] = ExperimentStatus(exp_dict["status"])

                    # Parse dates
                    for date_field in ("created_at", "updated_at", "started_at", "completed_at", "start_date", "end_date"):
                        if exp_dict.get(date_field):
                            exp_dict[date_field] = datetime.fromisoformat(exp_dict[date_field])

                    # Parse variants
                    variants = []
                    for v in exp_dict.pop("variants", []):
                        variants.append(ExperimentVariant(**v))
                    exp_dict["variants"] = variants

                    # Parse metrics
                    metrics = []
                    for m in exp_dict.pop("metrics", []):
                        m["metric_type"] = MetricType(m["metric_type"])
                        metrics.append(ExperimentMetric(**m))
                    exp_dict["metrics"] = metrics

                    experiment = Experiment(**exp_dict)

                    async with self._lock:
                        self._experiments[experiment.id] = experiment

                except Exception as e:
                    logger.warning(f"Failed to load experiment {exp_id}: {e}")

        logger.info(f"Loaded {len(self._experiments)} experiments")

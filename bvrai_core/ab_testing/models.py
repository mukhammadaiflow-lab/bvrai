"""A/B testing data models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


class ExperimentStatus(str, Enum):
    """Experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELED = "canceled"


class MetricType(str, Enum):
    """Types of metrics to track."""
    SUCCESS_RATE = "success_rate"
    AVG_DURATION = "avg_duration"
    SENTIMENT_SCORE = "sentiment_score"
    CONVERSION_RATE = "conversion_rate"
    CALL_COMPLETION_RATE = "call_completion_rate"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    TRANSFER_RATE = "transfer_rate"
    FIRST_CALL_RESOLUTION = "first_call_resolution"
    COST_PER_CALL = "cost_per_call"
    CUSTOM = "custom"


class StatisticalSignificance(str, Enum):
    """Statistical significance level."""
    NOT_SIGNIFICANT = "not_significant"
    LOW = "low"  # p < 0.1
    MEDIUM = "medium"  # p < 0.05
    HIGH = "high"  # p < 0.01


@dataclass
class ExperimentVariant:
    """A variant in an A/B test."""
    id: str
    name: str
    description: str = ""
    agent_id: str = ""  # Agent to use for this variant
    traffic_percentage: float = 50.0
    is_control: bool = False
    agent_config_overrides: Dict[str, Any] = field(default_factory=dict)

    # Metrics (populated during experiment)
    total_calls: int = 0
    successful_calls: int = 0
    total_duration: int = 0
    total_sentiment: float = 0.0
    conversions: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    @property
    def avg_duration(self) -> float:
        if self.successful_calls == 0:
            return 0.0
        return self.total_duration / self.successful_calls

    @property
    def avg_sentiment(self) -> float:
        if self.successful_calls == 0:
            return 0.0
        return self.total_sentiment / self.successful_calls

    @property
    def conversion_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return (self.conversions / self.total_calls) * 100


@dataclass
class ExperimentMetric:
    """Metric configuration for an experiment."""
    metric_type: MetricType
    name: str
    description: str = ""
    is_primary: bool = False
    target_value: Optional[float] = None
    minimum_improvement: float = 0.0  # e.g., 5% improvement needed
    custom_calculation: Optional[str] = None  # For custom metrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "metric_type": self.metric_type.value,
        }


@dataclass
class VariantAssignment:
    """Record of a user/call being assigned to a variant."""
    id: str
    experiment_id: str
    variant_id: str
    call_id: str
    contact_id: Optional[str] = None
    phone_number: str = ""
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "assigned_at": self.assigned_at.isoformat(),
        }


@dataclass
class VariantMetricResult:
    """Metric results for a variant."""
    variant_id: str
    variant_name: str
    metric_name: str
    value: float
    sample_size: int
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0
    standard_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricComparison:
    """Comparison between control and treatment for a metric."""
    metric_name: str
    control_value: float
    treatment_value: float
    absolute_lift: float
    relative_lift: float
    p_value: float
    significance: StatisticalSignificance
    confidence_interval: tuple = (0.0, 0.0)
    is_winner: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "significance": self.significance.value,
        }


@dataclass
class ExperimentResult:
    """Full results of an experiment."""
    experiment_id: str
    status: ExperimentStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_participants: int = 0
    variant_results: List[VariantMetricResult] = field(default_factory=list)
    metric_comparisons: List[MetricComparison] = field(default_factory=list)
    winning_variant_id: Optional[str] = None
    winning_variant_name: Optional[str] = None
    recommendation: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_participants": self.total_participants,
            "variant_results": [v.to_dict() for v in self.variant_results],
            "metric_comparisons": [m.to_dict() for m in self.metric_comparisons],
            "winning_variant_id": self.winning_variant_id,
            "winning_variant_name": self.winning_variant_name,
            "recommendation": self.recommendation,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class Experiment:
    """An A/B test experiment."""
    id: str
    organization_id: str
    name: str
    description: str = ""
    hypothesis: str = ""
    status: ExperimentStatus = ExperimentStatus.DRAFT

    # Variants
    variants: List[ExperimentVariant] = field(default_factory=list)

    # Metrics to track
    metrics: List[ExperimentMetric] = field(default_factory=list)
    primary_metric: Optional[str] = None

    # Targeting
    phone_number_ids: List[str] = field(default_factory=list)  # Which phone numbers to include
    agent_ids: List[str] = field(default_factory=list)  # Base agent(s) being tested
    traffic_percentage: float = 100.0  # % of eligible traffic to include

    # Duration
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_sample_size: int = 100
    max_sample_size: Optional[int] = None

    # Statistical settings
    confidence_level: float = 0.95  # 95% confidence
    minimum_detectable_effect: float = 0.05  # 5% MDE

    # Results
    total_participants: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metadata
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "status": self.status.value,
            "variants": [v.to_dict() for v in self.variants],
            "metrics": [m.to_dict() for m in self.metrics],
            "primary_metric": self.primary_metric,
            "phone_number_ids": self.phone_number_ids,
            "agent_ids": self.agent_ids,
            "traffic_percentage": self.traffic_percentage,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "min_sample_size": self.min_sample_size,
            "max_sample_size": self.max_sample_size,
            "confidence_level": self.confidence_level,
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "total_participants": self.total_participants,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_by": self.created_by,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @property
    def is_active(self) -> bool:
        return self.status == ExperimentStatus.RUNNING

    @property
    def control_variant(self) -> Optional[ExperimentVariant]:
        for v in self.variants:
            if v.is_control:
                return v
        return self.variants[0] if self.variants else None

    @property
    def treatment_variants(self) -> List[ExperimentVariant]:
        return [v for v in self.variants if not v.is_control]

    def validate(self) -> List[str]:
        """Validate experiment configuration."""
        errors = []

        if len(self.variants) < 2:
            errors.append("Experiment must have at least 2 variants")

        # Check traffic percentages sum to 100
        total_traffic = sum(v.traffic_percentage for v in self.variants)
        if abs(total_traffic - 100.0) > 0.01:
            errors.append(f"Variant traffic percentages must sum to 100 (got {total_traffic})")

        # Check for exactly one control
        controls = [v for v in self.variants if v.is_control]
        if len(controls) != 1:
            errors.append("Experiment must have exactly one control variant")

        # Check metrics
        if not self.metrics:
            errors.append("Experiment must have at least one metric")

        primary_metrics = [m for m in self.metrics if m.is_primary]
        if len(primary_metrics) != 1:
            errors.append("Experiment must have exactly one primary metric")

        return errors

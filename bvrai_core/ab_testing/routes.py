"""API routes for A/B testing system."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from .models import (
    Experiment,
    ExperimentStatus,
    ExperimentResult,
    MetricType,
)
from .manager import ExperimentManager
from .analyzer import ExperimentAnalyzer
from .router import VariantRouter, BanditRouter


router = APIRouter(prefix="/ab-testing", tags=["A/B Testing"])

# Dependency injection placeholders
_experiment_manager: Optional[ExperimentManager] = None
_experiment_analyzer: Optional[ExperimentAnalyzer] = None
_variant_router: Optional[VariantRouter] = None
_bandit_router: Optional[BanditRouter] = None


def get_experiment_manager() -> ExperimentManager:
    if not _experiment_manager:
        raise HTTPException(status_code=503, detail="Experiment manager not initialized")
    return _experiment_manager


def get_experiment_analyzer() -> ExperimentAnalyzer:
    if not _experiment_analyzer:
        raise HTTPException(status_code=503, detail="Experiment analyzer not initialized")
    return _experiment_analyzer


def get_variant_router() -> VariantRouter:
    if not _variant_router:
        raise HTTPException(status_code=503, detail="Variant router not initialized")
    return _variant_router


def init_routes(
    manager: ExperimentManager,
    analyzer: ExperimentAnalyzer,
    variant_router: VariantRouter,
    bandit_router: Optional[BanditRouter] = None,
) -> None:
    """Initialize route dependencies."""
    global _experiment_manager, _experiment_analyzer, _variant_router, _bandit_router
    _experiment_manager = manager
    _experiment_analyzer = analyzer
    _variant_router = variant_router
    _bandit_router = bandit_router


# Request/Response Models

class VariantCreate(BaseModel):
    """Request model for creating a variant."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    agent_id: str = ""
    traffic_percentage: float = Field(50.0, ge=0, le=100)
    is_control: bool = False
    agent_config_overrides: Dict[str, Any] = Field(default_factory=dict)


class MetricCreate(BaseModel):
    """Request model for creating a metric."""
    metric_type: str = Field(..., description="Type of metric to track")
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    is_primary: bool = False
    target_value: Optional[float] = None
    minimum_improvement: float = 0.0


class ExperimentCreate(BaseModel):
    """Request model for creating an experiment."""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    hypothesis: str = ""
    variants: List[VariantCreate] = Field(..., min_items=2)
    metrics: List[MetricCreate] = Field(..., min_items=1)
    phone_number_ids: List[str] = Field(default_factory=list)
    agent_ids: List[str] = Field(default_factory=list)
    traffic_percentage: float = Field(100.0, ge=0, le=100)
    min_sample_size: int = Field(100, ge=10)
    confidence_level: float = Field(0.95, ge=0.8, le=0.99)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentUpdate(BaseModel):
    """Request model for updating an experiment."""
    name: Optional[str] = None
    description: Optional[str] = None
    hypothesis: Optional[str] = None
    traffic_percentage: Optional[float] = None
    min_sample_size: Optional[int] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class VariantUpdate(BaseModel):
    """Request model for updating a variant."""
    name: Optional[str] = None
    description: Optional[str] = None
    agent_id: Optional[str] = None
    traffic_percentage: Optional[float] = None
    agent_config_overrides: Optional[Dict[str, Any]] = None


class VariantAssignmentRequest(BaseModel):
    """Request model for variant assignment."""
    phone_number: str = ""
    agent_id: str = ""
    sticky: bool = True
    use_bandit: bool = False


class CallResultRecord(BaseModel):
    """Request model for recording call results."""
    variant_id: str
    call_id: str
    success: bool
    duration: int = Field(..., ge=0)
    sentiment_score: float = Field(0.0, ge=-1, le=1)
    converted: bool = False
    custom_metrics: Dict[str, float] = Field(default_factory=dict)


class SampleSizeRequest(BaseModel):
    """Request model for sample size calculation."""
    baseline_rate: float = Field(..., ge=0, le=1, description="Expected baseline rate (e.g., 0.10 for 10%)")
    minimum_detectable_effect: float = Field(..., ge=0.01, le=1, description="Minimum relative effect to detect")
    confidence_level: float = Field(0.95, ge=0.8, le=0.99)
    power: float = Field(0.80, ge=0.5, le=0.99)
    num_variants: int = Field(2, ge=2, le=10)


class ExperimentResponse(BaseModel):
    """Response model for experiment."""
    id: str
    organization_id: str
    name: str
    description: str
    hypothesis: str
    status: str
    variants: List[Dict[str, Any]]
    metrics: List[Dict[str, Any]]
    traffic_percentage: float
    min_sample_size: int
    confidence_level: float
    total_participants: int
    created_at: str
    updated_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    tags: List[str]

    @classmethod
    def from_experiment(cls, exp: Experiment) -> "ExperimentResponse":
        return cls(
            id=exp.id,
            organization_id=exp.organization_id,
            name=exp.name,
            description=exp.description,
            hypothesis=exp.hypothesis,
            status=exp.status.value,
            variants=[v.to_dict() for v in exp.variants],
            metrics=[m.to_dict() for m in exp.metrics],
            traffic_percentage=exp.traffic_percentage,
            min_sample_size=exp.min_sample_size,
            confidence_level=exp.confidence_level,
            total_participants=exp.total_participants,
            created_at=exp.created_at.isoformat(),
            updated_at=exp.updated_at.isoformat(),
            started_at=exp.started_at.isoformat() if exp.started_at else None,
            completed_at=exp.completed_at.isoformat() if exp.completed_at else None,
            tags=exp.tags,
        )


class ExperimentListResponse(BaseModel):
    """Response model for experiment list."""
    experiments: List[ExperimentResponse]
    total: int
    limit: int
    offset: int


class VariantAssignmentResponse(BaseModel):
    """Response model for variant assignment."""
    experiment_id: str
    experiment_name: str
    variant_id: str
    variant_name: str
    is_control: bool
    agent_config_overrides: Dict[str, Any]


class SampleSizeResponse(BaseModel):
    """Response model for sample size calculation."""
    required_sample_per_variant: int
    total_sample_size: int
    estimated_duration_days: Optional[int] = None


# Experiment CRUD Routes

@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    organization_id: str,
    request: ExperimentCreate,
    created_by: str = "",
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> ExperimentResponse:
    """Create a new A/B testing experiment."""
    try:
        experiment = await manager.create_experiment(
            organization_id=organization_id,
            name=request.name,
            description=request.description,
            hypothesis=request.hypothesis,
            variants=[v.model_dump() for v in request.variants],
            metrics=[m.model_dump() for m in request.metrics],
            phone_number_ids=request.phone_number_ids,
            agent_ids=request.agent_ids,
            traffic_percentage=request.traffic_percentage,
            min_sample_size=request.min_sample_size,
            confidence_level=request.confidence_level,
            created_by=created_by,
            tags=request.tags,
            metadata=request.metadata,
        )
        return ExperimentResponse.from_experiment(experiment)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/experiments", response_model=ExperimentListResponse)
async def list_experiments(
    organization_id: str,
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> ExperimentListResponse:
    """List experiments for an organization."""
    exp_status = ExperimentStatus(status) if status else None
    experiments, total = await manager.list_experiments(
        organization_id=organization_id,
        status=exp_status,
        limit=limit,
        offset=offset,
    )
    return ExperimentListResponse(
        experiments=[ExperimentResponse.from_experiment(e) for e in experiments],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> ExperimentResponse:
    """Get an experiment by ID."""
    experiment = await manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentResponse.from_experiment(experiment)


@router.patch("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(
    experiment_id: str,
    request: ExperimentUpdate,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> ExperimentResponse:
    """Update an experiment (only draft experiments can be updated)."""
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    try:
        experiment = await manager.update_experiment(experiment_id, updates)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return ExperimentResponse.from_experiment(experiment)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/experiments/{experiment_id}", status_code=204)
async def delete_experiment(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> None:
    """Delete an experiment (cannot delete running experiments)."""
    try:
        deleted = await manager.delete_experiment(experiment_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Experiment not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Experiment Lifecycle Routes

@router.post("/experiments/{experiment_id}/start", response_model=ExperimentResponse)
async def start_experiment(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> ExperimentResponse:
    """Start an experiment."""
    try:
        experiment = await manager.start_experiment(experiment_id)
        return ExperimentResponse.from_experiment(experiment)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/experiments/{experiment_id}/pause", response_model=ExperimentResponse)
async def pause_experiment(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> ExperimentResponse:
    """Pause a running experiment."""
    try:
        experiment = await manager.pause_experiment(experiment_id)
        return ExperimentResponse.from_experiment(experiment)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/experiments/{experiment_id}/complete", response_model=ExperimentResponse)
async def complete_experiment(
    experiment_id: str,
    winning_variant_id: Optional[str] = None,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> ExperimentResponse:
    """Complete an experiment."""
    try:
        experiment = await manager.complete_experiment(experiment_id, winning_variant_id)
        return ExperimentResponse.from_experiment(experiment)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/experiments/{experiment_id}/cancel", response_model=ExperimentResponse)
async def cancel_experiment(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> ExperimentResponse:
    """Cancel an experiment."""
    try:
        experiment = await manager.cancel_experiment(experiment_id)
        return ExperimentResponse.from_experiment(experiment)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Variant Management Routes

@router.post("/experiments/{experiment_id}/variants", response_model=Dict[str, Any], status_code=201)
async def add_variant(
    experiment_id: str,
    request: VariantCreate,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> Dict[str, Any]:
    """Add a variant to a draft experiment."""
    try:
        variant = await manager.add_variant(experiment_id, request.model_dump())
        return variant.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/experiments/{experiment_id}/variants/{variant_id}", response_model=Dict[str, Any])
async def update_variant(
    experiment_id: str,
    variant_id: str,
    request: VariantUpdate,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> Dict[str, Any]:
    """Update a variant in a draft experiment."""
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    try:
        variant = await manager.update_variant(experiment_id, variant_id, updates)
        if not variant:
            raise HTTPException(status_code=404, detail="Variant not found")
        return variant.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/experiments/{experiment_id}/variants/{variant_id}", status_code=204)
async def remove_variant(
    experiment_id: str,
    variant_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> None:
    """Remove a variant from a draft experiment."""
    try:
        removed = await manager.remove_variant(experiment_id, variant_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Variant not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Variant Assignment Routes

@router.post("/experiments/{experiment_id}/assign", response_model=VariantAssignmentResponse)
async def assign_variant(
    experiment_id: str,
    request: VariantAssignmentRequest,
    manager: ExperimentManager = Depends(get_experiment_manager),
    variant_router: VariantRouter = Depends(get_variant_router),
) -> VariantAssignmentResponse:
    """Assign a variant to a call/user for an experiment."""
    experiment = await manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if not experiment.is_active:
        raise HTTPException(status_code=400, detail="Experiment is not running")

    # Use bandit router if requested and available
    if request.use_bandit and _bandit_router:
        variant = _bandit_router.select_variant_adaptive(experiment, request.phone_number)
    else:
        variant = variant_router.select_variant(
            experiment,
            request.phone_number,
            request.sticky,
        )

    if not variant:
        raise HTTPException(status_code=400, detail="No variant available (call may be excluded from experiment)")

    return VariantAssignmentResponse(
        experiment_id=experiment.id,
        experiment_name=experiment.name,
        variant_id=variant.id,
        variant_name=variant.name,
        is_control=variant.is_control,
        agent_config_overrides=variant.agent_config_overrides,
    )


@router.post("/experiments/{experiment_id}/assignments/bulk", response_model=List[VariantAssignmentResponse])
async def bulk_assign_variants(
    experiment_id: str,
    phone_numbers: List[str],
    manager: ExperimentManager = Depends(get_experiment_manager),
    variant_router: VariantRouter = Depends(get_variant_router),
) -> List[VariantAssignmentResponse]:
    """Assign variants to multiple phone numbers."""
    experiment = await manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if not experiment.is_active:
        raise HTTPException(status_code=400, detail="Experiment is not running")

    assignments = []
    for phone in phone_numbers:
        variant = variant_router.select_variant(experiment, phone, sticky=True)
        if variant:
            assignments.append(VariantAssignmentResponse(
                experiment_id=experiment.id,
                experiment_name=experiment.name,
                variant_id=variant.id,
                variant_name=variant.name,
                is_control=variant.is_control,
                agent_config_overrides=variant.agent_config_overrides,
            ))

    return assignments


# Result Recording Routes

@router.post("/experiments/{experiment_id}/results", status_code=201)
async def record_call_result(
    experiment_id: str,
    request: CallResultRecord,
    background_tasks: BackgroundTasks,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> Dict[str, str]:
    """Record call result for experiment metrics."""
    experiment = await manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Record result in background
    background_tasks.add_task(
        manager.record_call_result,
        experiment_id=experiment_id,
        variant_id=request.variant_id,
        call_id=request.call_id,
        success=request.success,
        duration=request.duration,
        sentiment_score=request.sentiment_score,
        converted=request.converted,
        custom_metrics=request.custom_metrics,
    )

    # Record for bandit learning if using adaptive allocation
    if _bandit_router:
        background_tasks.add_task(
            _bandit_router.record_outcome,
            experiment_id=experiment_id,
            variant_id=request.variant_id,
            success=request.success,
        )

    return {"status": "recorded"}


@router.post("/experiments/{experiment_id}/assignments/{call_id}", status_code=201)
async def record_assignment(
    experiment_id: str,
    call_id: str,
    variant_id: str,
    phone_number: str = "",
    contact_id: Optional[str] = None,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> Dict[str, Any]:
    """Record a call assignment to a variant."""
    assignment = await manager.record_assignment(
        experiment_id=experiment_id,
        variant_id=variant_id,
        call_id=call_id,
        phone_number=phone_number,
        contact_id=contact_id,
    )
    return assignment.to_dict()


# Analysis Routes

@router.get("/experiments/{experiment_id}/results", response_model=Dict[str, Any])
async def get_experiment_results(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager),
    analyzer: ExperimentAnalyzer = Depends(get_experiment_analyzer),
) -> Dict[str, Any]:
    """Get full analysis results for an experiment."""
    experiment = await manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    result = analyzer.analyze_experiment(experiment)
    return result.to_dict()


@router.get("/experiments/{experiment_id}/metrics", response_model=Dict[str, Any])
async def get_experiment_metrics(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> Dict[str, Any]:
    """Get current metrics for an experiment."""
    experiment = await manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    variants_metrics = []
    for variant in experiment.variants:
        variants_metrics.append({
            "variant_id": variant.id,
            "variant_name": variant.name,
            "is_control": variant.is_control,
            "total_calls": variant.total_calls,
            "successful_calls": variant.successful_calls,
            "success_rate": variant.success_rate,
            "avg_duration": variant.avg_duration,
            "avg_sentiment": variant.avg_sentiment,
            "conversion_rate": variant.conversion_rate,
            "conversions": variant.conversions,
            "custom_metrics": variant.custom_metrics,
        })

    return {
        "experiment_id": experiment.id,
        "experiment_name": experiment.name,
        "status": experiment.status.value,
        "total_participants": experiment.total_participants,
        "variants": variants_metrics,
    }


@router.get("/experiments/{experiment_id}/significance", response_model=Dict[str, Any])
async def check_statistical_significance(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager),
    analyzer: ExperimentAnalyzer = Depends(get_experiment_analyzer),
) -> Dict[str, Any]:
    """Check if experiment has reached statistical significance."""
    experiment = await manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    result = analyzer.analyze_experiment(experiment)

    significant_metrics = []
    for comparison in result.metric_comparisons:
        if comparison.significance.value in ("high", "medium"):
            significant_metrics.append({
                "metric_name": comparison.metric_name,
                "significance": comparison.significance.value,
                "p_value": comparison.p_value,
                "relative_lift": comparison.relative_lift,
                "is_winner": comparison.is_winner,
            })

    return {
        "experiment_id": experiment.id,
        "has_winner": result.winning_variant_id is not None,
        "winning_variant_id": result.winning_variant_id,
        "winning_variant_name": result.winning_variant_name,
        "significant_metrics": significant_metrics,
        "recommendation": result.recommendation,
        "sample_size_reached": experiment.total_participants >= experiment.min_sample_size,
        "current_sample_size": experiment.total_participants,
        "required_sample_size": experiment.min_sample_size,
    }


# Utility Routes

@router.post("/calculate-sample-size", response_model=SampleSizeResponse)
async def calculate_sample_size(
    request: SampleSizeRequest,
    daily_traffic: Optional[int] = None,
    traffic_percentage: float = 100.0,
    analyzer: ExperimentAnalyzer = Depends(get_experiment_analyzer),
) -> SampleSizeResponse:
    """Calculate required sample size for an experiment."""
    required_per_variant = analyzer.calculate_required_sample_size(
        baseline_rate=request.baseline_rate,
        minimum_detectable_effect=request.minimum_detectable_effect,
        confidence_level=request.confidence_level,
        power=request.power,
        num_variants=request.num_variants,
    )

    total_sample = required_per_variant * request.num_variants

    duration_days = None
    if daily_traffic:
        duration_days = analyzer.calculate_experiment_duration(
            required_sample_size=required_per_variant,
            daily_traffic=daily_traffic,
            traffic_percentage=traffic_percentage,
            num_variants=request.num_variants,
        )

    return SampleSizeResponse(
        required_sample_per_variant=required_per_variant,
        total_sample_size=total_sample,
        estimated_duration_days=duration_days,
    )


@router.get("/active-experiments", response_model=List[ExperimentResponse])
async def get_active_experiments(
    organization_id: str,
    phone_number_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> List[ExperimentResponse]:
    """Get active experiments for a phone number or agent."""
    if phone_number_id:
        experiments = await manager.get_active_experiments_for_phone(
            organization_id, phone_number_id
        )
    elif agent_id:
        experiments = await manager.get_active_experiments_for_agent(
            organization_id, agent_id
        )
    else:
        experiments, _ = await manager.list_experiments(
            organization_id=organization_id,
            status=ExperimentStatus.RUNNING,
        )

    return [ExperimentResponse.from_experiment(e) for e in experiments]


@router.get("/experiments/{experiment_id}/bandit-probabilities", response_model=Dict[str, float])
async def get_bandit_probabilities(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager),
) -> Dict[str, float]:
    """Get current probability estimates for bandit-based experiments."""
    if not _bandit_router:
        raise HTTPException(status_code=400, detail="Bandit router not enabled")

    experiment = await manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return _bandit_router.get_variant_probabilities(experiment)

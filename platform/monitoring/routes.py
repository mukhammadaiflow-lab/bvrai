"""API routes for real-time call monitoring dashboard."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel, Field

from .dashboard import CallMonitorDashboard, CallStatus
from .metrics import MetricsCollector, TimeGranularity
from .alerts import AlertManager, AlertRule, AlertCondition, AlertSeverity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


# Request/Response models

class ActiveCallResponse(BaseModel):
    """Active call response model."""
    id: str
    agent_id: str
    agent_name: str
    to_number: str
    from_number: str
    direction: str
    status: str
    started_at: str
    duration: int
    current_sentiment: Optional[str] = None
    sentiment_score: float = 0.0
    transcript_preview: str = ""
    is_speaking: bool = False


class DashboardMetricsResponse(BaseModel):
    """Dashboard metrics response model."""
    active_calls: int
    calls_today: int
    calls_this_hour: int
    avg_duration: float
    success_rate: float
    queued_calls: int
    agents_in_use: int
    total_agents: int
    concurrent_limit: int
    current_concurrency: int
    error_rate: float
    avg_wait_time: float
    sentiment_breakdown: Dict[str, int]


class AlertRuleCreate(BaseModel):
    """Create alert rule request."""
    name: str
    description: str = ""
    metric_name: str
    condition: str
    threshold: float
    comparison_period: int = 300
    severity: str = "warning"
    labels: Dict[str, str] = Field(default_factory=dict)
    cooldown_period: int = 300
    notification_channels: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class AlertRuleUpdate(BaseModel):
    """Update alert rule request."""
    name: Optional[str] = None
    description: Optional[str] = None
    threshold: Optional[float] = None
    severity: Optional[str] = None
    enabled: Optional[bool] = None
    notification_channels: Optional[List[str]] = None


class AlertRuleResponse(BaseModel):
    """Alert rule response model."""
    id: str
    name: str
    description: str
    metric_name: str
    condition: str
    threshold: float
    comparison_period: int
    severity: str
    organization_id: str
    enabled: bool
    notification_channels: List[str]
    tags: List[str]


class AlertResponse(BaseModel):
    """Alert response model."""
    id: str
    rule_id: str
    rule_name: str
    severity: str
    status: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    triggered_at: str
    resolved_at: Optional[str] = None
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None


class MetricQueryParams(BaseModel):
    """Metric query parameters."""
    metric_name: str
    granularity: str = "minute"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    labels: Dict[str, str] = Field(default_factory=dict)


class MetricDataPoint(BaseModel):
    """Metric data point response."""
    timestamp: str
    value: float
    count: int = 0
    min: float = 0
    max: float = 0
    avg: float = 0
    p50: float = 0
    p95: float = 0
    p99: float = 0


class MetricResponse(BaseModel):
    """Metric response model."""
    name: str
    data: List[MetricDataPoint]


# Dependency injection (these would be replaced with actual DI in production)
_dashboard: Optional[CallMonitorDashboard] = None
_metrics: Optional[MetricsCollector] = None
_alerts: Optional[AlertManager] = None


def get_dashboard() -> CallMonitorDashboard:
    global _dashboard
    if _dashboard is None:
        _dashboard = CallMonitorDashboard()
    return _dashboard


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def get_alerts() -> AlertManager:
    global _alerts
    if _alerts is None:
        _alerts = AlertManager()
    return _alerts


# Dashboard routes

@router.get("/active-calls", response_model=List[ActiveCallResponse])
async def get_active_calls(
    organization_id: str = Query(..., description="Organization ID"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    dashboard: CallMonitorDashboard = Depends(get_dashboard),
):
    """Get all active calls for an organization."""
    calls = await dashboard.get_active_calls(organization_id)

    if agent_id:
        calls = [c for c in calls if c.agent_id == agent_id]

    if status:
        calls = [c for c in calls if c.status.value == status]

    return [
        ActiveCallResponse(
            id=c.id,
            agent_id=c.agent_id,
            agent_name=c.agent_name,
            to_number=c.to_number,
            from_number=c.from_number,
            direction=c.direction,
            status=c.status.value,
            started_at=c.started_at.isoformat(),
            duration=c.duration,
            current_sentiment=c.current_sentiment,
            sentiment_score=c.sentiment_score,
            transcript_preview=c.transcript_preview,
            is_speaking=c.is_speaking,
        )
        for c in calls
    ]


@router.get("/active-calls/{call_id}", response_model=ActiveCallResponse)
async def get_active_call(
    call_id: str,
    dashboard: CallMonitorDashboard = Depends(get_dashboard),
):
    """Get details for a specific active call."""
    async with dashboard._lock:
        call = dashboard._active_calls.get(call_id)

    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    return ActiveCallResponse(
        id=call.id,
        agent_id=call.agent_id,
        agent_name=call.agent_name,
        to_number=call.to_number,
        from_number=call.from_number,
        direction=call.direction,
        status=call.status.value,
        started_at=call.started_at.isoformat(),
        duration=call.duration,
        current_sentiment=call.current_sentiment,
        sentiment_score=call.sentiment_score,
        transcript_preview=call.transcript_preview,
        is_speaking=call.is_speaking,
    )


@router.get("/dashboard/metrics", response_model=DashboardMetricsResponse)
async def get_dashboard_metrics(
    organization_id: str = Query(..., description="Organization ID"),
    dashboard: CallMonitorDashboard = Depends(get_dashboard),
):
    """Get dashboard metrics for an organization."""
    metrics = await dashboard.get_metrics(organization_id)
    return DashboardMetricsResponse(**metrics.to_dict())


@router.get("/active-calls/{call_id}/transcript")
async def get_call_transcript(
    call_id: str,
    dashboard: CallMonitorDashboard = Depends(get_dashboard),
):
    """Get transcript for an active call."""
    transcript = await dashboard._get_call_transcript(call_id)
    return {"call_id": call_id, "messages": transcript}


# WebSocket endpoint for real-time updates

@router.websocket("/ws/{organization_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    organization_id: str,
    dashboard: CallMonitorDashboard = Depends(get_dashboard),
):
    """WebSocket endpoint for real-time dashboard updates."""
    # In production, validate the token/user
    user_id = "anonymous"

    await dashboard.handle_websocket(websocket, organization_id, user_id)


# Metrics routes

@router.get("/metrics/overview")
async def get_metrics_overview(
    organization_id: str = Query(..., description="Organization ID"),
    period: str = Query("hour", description="Time period (hour, day, week, month)"),
    collector: MetricsCollector = Depends(get_metrics),
):
    """Get metrics overview for an organization."""
    return await collector.get_organization_metrics(organization_id, period)


@router.post("/metrics/query", response_model=MetricResponse)
async def query_metrics(
    query: MetricQueryParams,
    organization_id: str = Query(..., description="Organization ID"),
    collector: MetricsCollector = Depends(get_metrics),
):
    """Query metrics with custom parameters."""
    granularity = TimeGranularity(query.granularity)

    end_time = datetime.utcnow()
    if query.end_time:
        end_time = datetime.fromisoformat(query.end_time)

    start_time = end_time - timedelta(hours=1)
    if query.start_time:
        start_time = datetime.fromisoformat(query.start_time)

    labels = {"organization_id": organization_id, **query.labels}

    results = await collector.get_metric(
        query.metric_name,
        labels,
        granularity,
        start_time,
        end_time,
    )

    return MetricResponse(
        name=query.metric_name,
        data=[
            MetricDataPoint(
                timestamp=r.period_start.isoformat(),
                value=r.avg,
                count=r.count,
                min=r.min,
                max=r.max,
                avg=r.avg,
                p50=r.p50,
                p95=r.p95,
                p99=r.p99,
            )
            for r in results
        ],
    )


@router.get("/metrics/realtime")
async def get_realtime_metrics(
    organization_id: str = Query(..., description="Organization ID"),
    collector: MetricsCollector = Depends(get_metrics),
):
    """Get real-time metrics (gauges and recent counters)."""
    labels = {"organization_id": organization_id}

    # Get current gauges
    active_calls = await collector.get_gauge_value("calls_active", labels)
    concurrent = await collector.get_gauge_value("concurrent_calls", labels)
    queue_length = await collector.get_gauge_value("queue_length", labels)

    # Get histogram stats
    duration_stats = await collector.get_histogram_stats("call_duration_seconds", labels)
    llm_latency_stats = await collector.get_histogram_stats("llm_response_time_ms", labels)

    return {
        "active_calls": active_calls,
        "concurrent_calls": concurrent,
        "queue_length": queue_length,
        "call_duration": duration_stats,
        "llm_latency": llm_latency_stats,
    }


# Alert routes

@router.get("/alerts", response_model=List[AlertResponse])
async def get_active_alerts(
    organization_id: str = Query(..., description="Organization ID"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    alert_manager: AlertManager = Depends(get_alerts),
):
    """Get active alerts for an organization."""
    sev = AlertSeverity(severity) if severity else None
    alerts = await alert_manager.get_active_alerts(organization_id, sev)

    return [
        AlertResponse(
            id=a.id,
            rule_id=a.rule_id,
            rule_name=a.rule_name,
            severity=a.severity.value,
            status=a.status.value,
            message=a.message,
            metric_name=a.metric_name,
            metric_value=a.metric_value,
            threshold=a.threshold,
            triggered_at=a.triggered_at.isoformat(),
            resolved_at=a.resolved_at.isoformat() if a.resolved_at else None,
            acknowledged_at=a.acknowledged_at.isoformat() if a.acknowledged_at else None,
            acknowledged_by=a.acknowledged_by,
        )
        for a in alerts
    ]


@router.get("/alerts/history", response_model=List[AlertResponse])
async def get_alert_history(
    organization_id: str = Query(..., description="Organization ID"),
    limit: int = Query(100, ge=1, le=1000),
    start_time: Optional[str] = Query(None, description="Start time (ISO 8601)"),
    end_time: Optional[str] = Query(None, description="End time (ISO 8601)"),
    alert_manager: AlertManager = Depends(get_alerts),
):
    """Get alert history for an organization."""
    start = datetime.fromisoformat(start_time) if start_time else None
    end = datetime.fromisoformat(end_time) if end_time else None

    alerts = await alert_manager.get_alert_history(organization_id, limit, start, end)

    return [
        AlertResponse(
            id=a.id,
            rule_id=a.rule_id,
            rule_name=a.rule_name,
            severity=a.severity.value,
            status=a.status.value,
            message=a.message,
            metric_name=a.metric_name,
            metric_value=a.metric_value,
            threshold=a.threshold,
            triggered_at=a.triggered_at.isoformat(),
            resolved_at=a.resolved_at.isoformat() if a.resolved_at else None,
            acknowledged_at=a.acknowledged_at.isoformat() if a.acknowledged_at else None,
            acknowledged_by=a.acknowledged_by,
        )
        for a in alerts
    ]


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    user_id: str = Query(..., description="User ID acknowledging the alert"),
    alert_manager: AlertManager = Depends(get_alerts),
):
    """Acknowledge an active alert."""
    alert = await alert_manager.acknowledge_alert(alert_id, user_id)

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"status": "acknowledged", "alert_id": alert_id}


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    alert_manager: AlertManager = Depends(get_alerts),
):
    """Manually resolve an active alert."""
    alert = await alert_manager.resolve_alert(alert_id)

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"status": "resolved", "alert_id": alert_id}


# Alert rules routes

@router.get("/alert-rules", response_model=List[AlertRuleResponse])
async def list_alert_rules(
    organization_id: str = Query(..., description="Organization ID"),
    alert_manager: AlertManager = Depends(get_alerts),
):
    """List alert rules for an organization."""
    rules = await alert_manager.list_rules(organization_id)

    return [
        AlertRuleResponse(
            id=r.id,
            name=r.name,
            description=r.description,
            metric_name=r.metric_name,
            condition=r.condition.value,
            threshold=r.threshold,
            comparison_period=r.comparison_period,
            severity=r.severity.value,
            organization_id=r.organization_id,
            enabled=r.enabled,
            notification_channels=r.notification_channels,
            tags=r.tags,
        )
        for r in rules
    ]


@router.post("/alert-rules", response_model=AlertRuleResponse)
async def create_alert_rule(
    rule_data: AlertRuleCreate,
    organization_id: str = Query(..., description="Organization ID"),
    alert_manager: AlertManager = Depends(get_alerts),
):
    """Create a new alert rule."""
    rule = AlertRule(
        id=str(uuid4())[:8],
        name=rule_data.name,
        description=rule_data.description,
        metric_name=rule_data.metric_name,
        condition=AlertCondition(rule_data.condition),
        threshold=rule_data.threshold,
        comparison_period=rule_data.comparison_period,
        severity=AlertSeverity(rule_data.severity),
        organization_id=organization_id,
        labels=rule_data.labels,
        cooldown_period=rule_data.cooldown_period,
        notification_channels=rule_data.notification_channels,
        tags=rule_data.tags,
    )

    created = await alert_manager.create_rule(rule)

    return AlertRuleResponse(
        id=created.id,
        name=created.name,
        description=created.description,
        metric_name=created.metric_name,
        condition=created.condition.value,
        threshold=created.threshold,
        comparison_period=created.comparison_period,
        severity=created.severity.value,
        organization_id=created.organization_id,
        enabled=created.enabled,
        notification_channels=created.notification_channels,
        tags=created.tags,
    )


@router.patch("/alert-rules/{rule_id}", response_model=AlertRuleResponse)
async def update_alert_rule(
    rule_id: str,
    updates: AlertRuleUpdate,
    alert_manager: AlertManager = Depends(get_alerts),
):
    """Update an alert rule."""
    update_dict = updates.model_dump(exclude_unset=True)

    if "severity" in update_dict:
        update_dict["severity"] = AlertSeverity(update_dict["severity"])

    rule = await alert_manager.update_rule(rule_id, update_dict)

    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    return AlertRuleResponse(
        id=rule.id,
        name=rule.name,
        description=rule.description,
        metric_name=rule.metric_name,
        condition=rule.condition.value,
        threshold=rule.threshold,
        comparison_period=rule.comparison_period,
        severity=rule.severity.value,
        organization_id=rule.organization_id,
        enabled=rule.enabled,
        notification_channels=rule.notification_channels,
        tags=rule.tags,
    )


@router.delete("/alert-rules/{rule_id}")
async def delete_alert_rule(
    rule_id: str,
    alert_manager: AlertManager = Depends(get_alerts),
):
    """Delete an alert rule."""
    deleted = await alert_manager.delete_rule(rule_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Rule not found")

    return {"status": "deleted", "rule_id": rule_id}


@router.post("/alert-rules/{rule_id}/silence")
async def silence_alert_rule(
    rule_id: str,
    duration_seconds: int = Query(3600, description="Silence duration in seconds"),
    alert_manager: AlertManager = Depends(get_alerts),
):
    """Silence an alert rule for a period."""
    silenced = await alert_manager.silence_rule(rule_id, duration_seconds)

    if not silenced:
        raise HTTPException(status_code=404, detail="Rule not found")

    return {"status": "silenced", "rule_id": rule_id, "duration_seconds": duration_seconds}

"""
Analytics Engine Service.

Advanced analytics platform for voice AI applications.

API Endpoints:
- Events: Ingest analytics events
- Metrics: Query aggregated metrics
- Dashboards: Manage dashboards
- Reports: Generate reports
- Alerts: Manage and view alerts
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    MetricType,
    AggregationPeriod,
    EventType,
    get_settings,
)
from .models import (
    IngestEventRequest,
    IngestBatchRequest,
    QueryMetricsRequest,
    MetricsResponse,
    DashboardResponse,
    AlertsResponse,
    AnalyticsEvent,
)
from .collectors import EventCollector
from .aggregators import MetricsAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Service metadata
SERVICE_NAME = "analytics-engine"
SERVICE_VERSION = "1.0.0"
START_TIME = time.time()

# Global instances
collector: Optional[EventCollector] = None
aggregator: Optional[MetricsAggregator] = None

# In-memory storage (replace with database in production)
events_store: List[AnalyticsEvent] = []
metrics_cache: Dict[str, Any] = {}


def process_events(events: List[AnalyticsEvent]) -> None:
    """Process events callback."""
    global events_store, metrics_cache

    events_store.extend(events)

    # Keep last 10000 events in memory
    if len(events_store) > 10000:
        events_store = events_store[-10000:]

    # Update metrics cache
    period = AggregationPeriod.HOUR
    metrics = aggregator.aggregate_events(events, period)

    for metric in metrics:
        key = f"{metric.metric_type.value}:{metric.period.value}"
        metrics_cache[key] = metric.to_dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global collector, aggregator

    logger.info(f"Starting {SERVICE_NAME} v{SERVICE_VERSION}")

    # Initialize components
    collector = EventCollector()
    aggregator = MetricsAggregator()

    # Set processing callback
    collector.set_flush_callback(process_events)

    # Start collector
    await collector.start()

    logger.info("Analytics Engine initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Analytics Engine")
    await collector.stop()


# Create FastAPI app
app = FastAPI(
    title="Analytics Engine Service",
    description="Advanced analytics platform for voice AI applications",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Info
# =============================================================================


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "uptime_seconds": time.time() - START_TIME,
        "collector_metrics": collector.get_metrics() if collector else {},
    }


@app.get("/info")
async def get_info() -> Dict[str, Any]:
    """Get service information."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "metric_types": [m.value for m in MetricType],
        "event_types": [e.value for e in EventType],
        "aggregation_periods": [p.value for p in AggregationPeriod],
    }


# =============================================================================
# Events API
# =============================================================================


@app.post("/events")
async def ingest_event(request: IngestEventRequest) -> Dict[str, Any]:
    """Ingest a single analytics event."""
    success = await collector.collect_raw(
        event_type=request.event_type,
        data=request.data,
        organization_id=request.organization_id,
        agent_id=request.agent_id,
        call_id=request.call_id,
        session_id=request.session_id,
        timestamp=request.timestamp,
    )

    if not success:
        raise HTTPException(status_code=400, detail="Failed to ingest event")

    return {"status": "accepted", "event_type": request.event_type}


@app.post("/events/batch")
async def ingest_batch(request: IngestBatchRequest) -> Dict[str, Any]:
    """Ingest multiple analytics events."""
    accepted = 0
    for event in request.events:
        success = await collector.collect_raw(
            event_type=event.event_type,
            data=event.data,
            organization_id=event.organization_id,
            agent_id=event.agent_id,
            call_id=event.call_id,
            session_id=event.session_id,
            timestamp=event.timestamp,
        )
        if success:
            accepted += 1

    return {
        "status": "accepted",
        "total": len(request.events),
        "accepted": accepted,
        "rejected": len(request.events) - accepted,
    }


# =============================================================================
# Metrics API
# =============================================================================


@app.get("/metrics/{metric_type}")
async def get_metrics(
    metric_type: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    period: str = "hour",
    organization_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get aggregated metrics."""
    try:
        mtype = MetricType(metric_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown metric type: {metric_type}")

    try:
        agg_period = AggregationPeriod(period)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown period: {period}")

    # Default time range
    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        start_time = end_time - timedelta(hours=24)

    # Filter events
    filtered_events = events_store
    if organization_id:
        filtered_events = [e for e in filtered_events if e.organization_id == organization_id]
    if agent_id:
        filtered_events = [e for e in filtered_events if e.agent_id == agent_id]

    # Time filter
    filtered_events = [
        e for e in filtered_events
        if start_time <= e.timestamp <= end_time
    ]

    # Aggregate
    metrics = aggregator.aggregate_events(filtered_events, agg_period)

    # Filter by metric type
    metrics = [m for m in metrics if m.metric_type == mtype]

    return {
        "metric_type": metric_type,
        "period": period,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "data": [m.to_dict() for m in metrics],
        "summary": _calculate_summary(metrics),
    }


def _calculate_summary(metrics: list) -> Dict[str, Any]:
    """Calculate summary statistics."""
    if not metrics:
        return {"total_count": 0, "avg": 0, "max": 0, "min": 0}

    counts = [m.count for m in metrics]
    avgs = [m.avg for m in metrics]

    return {
        "total_count": sum(counts),
        "avg": sum(avgs) / len(avgs) if avgs else 0,
        "max": max(avgs) if avgs else 0,
        "min": min(avgs) if avgs else 0,
        "periods": len(metrics),
    }


@app.get("/metrics")
async def list_metrics() -> Dict[str, Any]:
    """List all available metrics with current values."""
    return {
        "metrics": [
            {
                "type": m.value,
                "cached": f"{m.value}:hour" in metrics_cache,
                "last_value": metrics_cache.get(f"{m.value}:hour", {}).get("avg", 0),
            }
            for m in MetricType
        ]
    }


# =============================================================================
# Real-Time Metrics
# =============================================================================


@app.get("/metrics/realtime/{metric_type}")
async def get_realtime_metrics(
    metric_type: str,
    window_seconds: int = Query(default=60, ge=10, le=3600),
) -> Dict[str, Any]:
    """Get real-time metrics."""
    try:
        mtype = MetricType(metric_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown metric type: {metric_type}")

    return aggregator.get_real_time_metrics(mtype, window_seconds)


# =============================================================================
# Dashboards API
# =============================================================================


# In-memory dashboard storage
DASHBOARDS = {
    "overview": {
        "dashboard_id": "overview",
        "name": "Overview",
        "widgets": [
            {
                "widget_id": "call-volume",
                "widget_type": "line_chart",
                "title": "Call Volume",
                "metrics": ["call_volume"],
            },
            {
                "widget_id": "latency",
                "widget_type": "gauge",
                "title": "Average Latency",
                "metrics": ["latency"],
            },
            {
                "widget_id": "sentiment",
                "widget_type": "pie_chart",
                "title": "Sentiment Distribution",
                "metrics": ["sentiment"],
            },
            {
                "widget_id": "conversions",
                "widget_type": "metric",
                "title": "Conversions",
                "metrics": ["conversion"],
            },
        ],
    },
    "performance": {
        "dashboard_id": "performance",
        "name": "Performance",
        "widgets": [
            {
                "widget_id": "latency-chart",
                "widget_type": "line_chart",
                "title": "Latency Over Time",
                "metrics": ["latency"],
            },
            {
                "widget_id": "error-rate",
                "widget_type": "line_chart",
                "title": "Error Rate",
                "metrics": ["error_rate"],
            },
            {
                "widget_id": "success-rate",
                "widget_type": "gauge",
                "title": "Success Rate",
                "metrics": ["success_rate"],
            },
        ],
    },
}


@app.get("/dashboards")
async def list_dashboards() -> Dict[str, Any]:
    """List available dashboards."""
    return {
        "dashboards": [
            {"id": d["dashboard_id"], "name": d["name"]}
            for d in DASHBOARDS.values()
        ]
    }


@app.get("/dashboards/{dashboard_id}")
async def get_dashboard(dashboard_id: str) -> Dict[str, Any]:
    """Get dashboard configuration with data."""
    if dashboard_id not in DASHBOARDS:
        raise HTTPException(status_code=404, detail="Dashboard not found")

    dashboard = DASHBOARDS[dashboard_id]

    # Populate widget data
    for widget in dashboard["widgets"]:
        widget["data"] = []
        for metric_type in widget["metrics"]:
            key = f"{metric_type}:hour"
            if key in metrics_cache:
                widget["data"].append(metrics_cache[key])

    return {
        **dashboard,
        "last_updated": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Alerts API
# =============================================================================


# In-memory alerts
ALERTS: List[Dict[str, Any]] = []


@app.get("/alerts")
async def list_alerts(
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> Dict[str, Any]:
    """List alerts."""
    alerts = ALERTS.copy()

    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]

    if acknowledged is not None:
        alerts = [a for a in alerts if a.get("acknowledged") == acknowledged]

    alerts = alerts[:limit]

    return {
        "alerts": alerts,
        "total": len(ALERTS),
        "unacknowledged": sum(1 for a in ALERTS if not a.get("acknowledged", False)),
    }


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str) -> Dict[str, Any]:
    """Acknowledge an alert."""
    for alert in ALERTS:
        if alert.get("alert_id") == alert_id:
            alert["acknowledged"] = True
            alert["acknowledged_at"] = datetime.utcnow().isoformat()
            return {"status": "acknowledged", "alert_id": alert_id}

    raise HTTPException(status_code=404, detail="Alert not found")


# =============================================================================
# Reports API
# =============================================================================


@app.get("/reports")
async def list_reports() -> Dict[str, Any]:
    """List available report types."""
    return {
        "reports": [
            {
                "id": "daily-summary",
                "name": "Daily Summary",
                "description": "Daily call and conversation summary",
            },
            {
                "id": "weekly-performance",
                "name": "Weekly Performance",
                "description": "Weekly performance metrics",
            },
            {
                "id": "agent-analysis",
                "name": "Agent Analysis",
                "description": "Per-agent performance breakdown",
            },
            {
                "id": "conversation-insights",
                "name": "Conversation Insights",
                "description": "Intent and sentiment analysis",
            },
        ]
    }


@app.post("/reports/{report_id}/generate")
async def generate_report(
    report_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Generate a report."""
    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        start_time = end_time - timedelta(days=1)

    # Filter events
    filtered_events = [
        e for e in events_store
        if start_time <= e.timestamp <= end_time
    ]

    # Aggregate
    metrics = aggregator.aggregate_events(filtered_events, AggregationPeriod.HOUR)

    return {
        "report_id": report_id,
        "generated_at": datetime.utcnow().isoformat(),
        "period": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        },
        "summary": {
            "total_events": len(filtered_events),
            "metric_count": len(metrics),
        },
        "data": [m.to_dict() for m in metrics[:100]],  # Limit for demo
    }


# =============================================================================
# WebSocket for Real-Time Updates
# =============================================================================


active_connections: List[WebSocket] = []


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """WebSocket for real-time analytics updates."""
    await websocket.accept()
    active_connections.append(websocket)

    logger.info("WebSocket client connected for live updates")

    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(5)

            update = {
                "type": "metrics_update",
                "timestamp": datetime.utcnow().isoformat(),
                "collector": collector.get_metrics() if collector else {},
                "recent_metrics": list(metrics_cache.values())[:10],
            }

            await websocket.send_json(update)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

    finally:
        active_connections.remove(websocket)


# =============================================================================
# Export API
# =============================================================================


@app.get("/export/events")
async def export_events(
    format: str = "json",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(default=1000, ge=1, le=10000),
) -> Dict[str, Any]:
    """Export events."""
    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        start_time = end_time - timedelta(hours=24)

    filtered = [
        e for e in events_store
        if start_time <= e.timestamp <= end_time
    ][:limit]

    return {
        "format": format,
        "count": len(filtered),
        "events": [e.to_dict() for e in filtered],
    }


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )

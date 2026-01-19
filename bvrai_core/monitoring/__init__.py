"""
Real-Time Monitoring & Alerting Module

This module provides comprehensive monitoring, alerting, health checking,
and incident management capabilities for the voice agent platform.

Components:
- Dashboard: Real-time call monitoring with WebSocket support
- Metrics: Call and system metrics collection with Redis backend
- Alerts: Alert management with multi-channel notifications
- Health: Health check system for services and dependencies
- Incidents: Incident lifecycle management with SLA tracking

Example usage:

    from bvrai_core.monitoring import (
        # Dashboard
        CallMonitorDashboard,
        # Metrics
        MetricsCollector,
        # Alerts
        AlertManager,
        # Health Checks
        HealthCheckManager,
        HTTPHealthChecker,
        TCPHealthChecker,
        DatabaseHealthChecker,
        RedisHealthChecker,
        # Incidents
        IncidentManager,
        # Types
        HealthStatus,
        AlertSeverity,
        IncidentSeverity,
        IncidentStatus,
        MetricValue,
        HealthCheck,
        Alert,
        Incident,
    )

    # Initialize monitoring components
    metrics = MetricsCollector(redis_client)
    alerts = AlertManager(redis_client)
    dashboard = CallMonitorDashboard(redis_client)

    # Set up health checks
    health_manager = HealthCheckManager()
    health_manager.register_checker(HTTPHealthChecker(
        name="api_service",
        url="http://localhost:8080/health",
        interval_seconds=30,
    ))
    health_manager.register_checker(DatabaseHealthChecker(
        name="postgres",
        connection_string="postgresql://...",
    ))

    # Start health monitoring
    await health_manager.start()
    status = await health_manager.get_status()

    # Incident management
    incident_mgr = IncidentManager()
    incident = await incident_mgr.create_incident(
        title="High error rate detected",
        severity=IncidentSeverity.HIGH,
        organization_id="org_123",
        description="Error rate exceeded 5% threshold",
        affected_services=["voice-gateway", "call-router"],
    )

    # Acknowledge and work on incident
    await incident_mgr.acknowledge_incident(
        incident_id=incident.id,
        commander="engineer@company.com",
    )
"""

# Dashboard - Real-time call monitoring
from .dashboard import CallMonitorDashboard

# Metrics - Call and system metrics collection
from .metrics import MetricsCollector

# Alerts - Alert management with notifications
from .alerts import AlertManager

# Base types and abstractions
from .base import (
    # Enums
    HealthStatus,
    AlertSeverity,
    IncidentSeverity,
    IncidentStatus,
    CheckType,
    MetricType,
    NotificationChannel,
    # Data classes
    MetricValue,
    HealthCheck,
    Alert,
    Incident,
    IncidentUpdate,
    Postmortem,
    SLAConfig,
    # Abstract interfaces
    HealthChecker,
    MetricCollector,
    NotificationSender,
)

# Health checking system
from .health import (
    HealthCheckManager,
    HTTPHealthChecker,
    TCPHealthChecker,
    DatabaseHealthChecker,
    RedisHealthChecker,
)

# Incident management
from .incidents import (
    IncidentManager,
    IncidentService,
)


__all__ = [
    # Dashboard
    "CallMonitorDashboard",
    # Metrics
    "MetricsCollector",
    # Alerts
    "AlertManager",
    # Health checking
    "HealthCheckManager",
    "HTTPHealthChecker",
    "TCPHealthChecker",
    "DatabaseHealthChecker",
    "RedisHealthChecker",
    # Incident management
    "IncidentManager",
    "IncidentService",
    # Enums
    "HealthStatus",
    "AlertSeverity",
    "IncidentSeverity",
    "IncidentStatus",
    "CheckType",
    "MetricType",
    "NotificationChannel",
    # Data classes
    "MetricValue",
    "HealthCheck",
    "Alert",
    "Incident",
    "IncidentUpdate",
    "Postmortem",
    "SLAConfig",
    # Abstract interfaces
    "HealthChecker",
    "MetricCollector",
    "NotificationSender",
]

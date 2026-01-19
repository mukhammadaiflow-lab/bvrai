"""Analytics module."""

from app.analytics.service import AnalyticsService
from app.analytics.routes import router

__all__ = ["AnalyticsService", "router"]

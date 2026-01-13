"""
Builder Engine Python SDK - Analytics Resource

This module provides methods for accessing analytics data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List

from builderengine.resources.base import BaseResource
from builderengine.models import Analytics, CallMetrics, UsageMetrics, Usage
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class AnalyticsResource(BaseResource):
    """
    Resource for accessing analytics data.

    Analytics provides insights into call performance, usage patterns,
    costs, and agent effectiveness. Use this resource to build dashboards
    and reports.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> overview = client.analytics.get_overview(period="month")
        >>> print(f"Total calls: {overview.call_metrics.total_calls}")
        >>> print(f"Success rate: {overview.call_metrics.success_rate:.1%}")
    """

    def get_overview(
        self,
        period: str = "week",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Analytics:
        """
        Get analytics overview.

        Args:
            period: Time period (day, week, month, quarter, year, custom)
            start_date: Start date for custom period (ISO format)
            end_date: End date for custom period (ISO format)
            agent_id: Filter by specific agent

        Returns:
            Analytics object with all metrics

        Example:
            >>> overview = client.analytics.get_overview(period="month")
            >>> print(f"Calls: {overview.call_metrics.total_calls}")
            >>> print(f"Duration: {overview.call_metrics.total_duration_seconds / 3600:.1f}h")
        """
        params = {
            "period": period,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if agent_id:
            params["agent_id"] = agent_id

        response = self._get(Endpoints.ANALYTICS_OVERVIEW, params=params)
        return Analytics.from_dict(response)

    def get_call_metrics(
        self,
        period: str = "week",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        agent_id: Optional[str] = None,
        group_by: str = "day",
    ) -> Dict[str, Any]:
        """
        Get call metrics over time.

        Args:
            period: Time period
            start_date: Start date for custom period
            end_date: End date for custom period
            agent_id: Filter by agent
            group_by: Grouping interval (hour, day, week, month)

        Returns:
            Call metrics with time series data

        Example:
            >>> metrics = client.analytics.get_call_metrics(
            ...     period="week",
            ...     group_by="day"
            ... )
            >>> for point in metrics["time_series"]:
            ...     print(f"{point['date']}: {point['total_calls']} calls")
        """
        params = {
            "period": period,
            "group_by": group_by,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if agent_id:
            params["agent_id"] = agent_id

        return self._get(Endpoints.ANALYTICS_CALLS, params=params)

    def get_agent_performance(
        self,
        period: str = "week",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sort_by: str = "total_calls",
        sort_order: str = "desc",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get agent performance rankings.

        Args:
            period: Time period
            start_date: Start date for custom period
            end_date: End date for custom period
            sort_by: Metric to sort by (total_calls, success_rate, avg_duration)
            sort_order: Sort order (asc, desc)
            limit: Number of agents to return

        Returns:
            List of agent performance data

        Example:
            >>> agents = client.analytics.get_agent_performance(
            ...     period="month",
            ...     sort_by="success_rate",
            ...     limit=5
            ... )
            >>> for agent in agents:
            ...     print(f"{agent['name']}: {agent['success_rate']:.1%}")
        """
        params = {
            "period": period,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "limit": limit,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        response = self._get(Endpoints.ANALYTICS_AGENTS, params=params)
        return response.get("agents", [])

    def get_usage(
        self,
        period: str = "month",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Usage:
        """
        Get usage statistics.

        Args:
            period: Time period
            start_date: Start date for custom period
            end_date: End date for custom period

        Returns:
            Usage object with current usage and limits
        """
        params = {"period": period}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        response = self._get(Endpoints.ANALYTICS_USAGE, params=params)
        return Usage.from_dict(response)

    def get_cost_breakdown(
        self,
        period: str = "month",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        group_by: str = "category",
    ) -> Dict[str, Any]:
        """
        Get cost breakdown.

        Args:
            period: Time period
            start_date: Start date for custom period
            end_date: End date for custom period
            group_by: Grouping (category, agent, day)

        Returns:
            Cost breakdown data

        Example:
            >>> costs = client.analytics.get_cost_breakdown(
            ...     period="month",
            ...     group_by="category"
            ... )
            >>> for category, amount in costs["breakdown"].items():
            ...     print(f"{category}: ${amount:.2f}")
        """
        params = {
            "period": period,
            "group_by": group_by,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return self._get(Endpoints.ANALYTICS_COSTS, params=params)

    def export(
        self,
        report_type: str,
        period: str = "month",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        format: str = "csv",
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export analytics data.

        Args:
            report_type: Type of report (calls, agents, usage, costs)
            period: Time period
            start_date: Start date for custom period
            end_date: End date for custom period
            format: Export format (csv, xlsx, json)
            agent_id: Filter by agent

        Returns:
            Export result with download URL

        Example:
            >>> export = client.analytics.export(
            ...     report_type="calls",
            ...     period="month",
            ...     format="csv"
            ... )
            >>> print(f"Download: {export['download_url']}")
        """
        params = {
            "report_type": report_type,
            "period": period,
            "format": format,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if agent_id:
            params["agent_id"] = agent_id

        return self._post(Endpoints.ANALYTICS_EXPORT, json=params)

    def get_sentiment_analysis(
        self,
        period: str = "week",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get sentiment analysis across calls.

        Args:
            period: Time period
            start_date: Start date for custom period
            end_date: End date for custom period
            agent_id: Filter by agent

        Returns:
            Sentiment distribution and trends
        """
        params = {"period": period}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if agent_id:
            params["agent_id"] = agent_id

        return self._get(f"{Endpoints.ANALYTICS_OVERVIEW}/sentiment", params=params)

    def get_top_intents(
        self,
        period: str = "week",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get most common intents detected in conversations.

        Args:
            period: Time period
            start_date: Start date for custom period
            end_date: End date for custom period
            agent_id: Filter by agent
            limit: Number of intents to return

        Returns:
            List of intents with counts and trends
        """
        params = {
            "period": period,
            "limit": limit,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if agent_id:
            params["agent_id"] = agent_id

        response = self._get(f"{Endpoints.ANALYTICS_OVERVIEW}/intents", params=params)
        return response.get("intents", [])

    def get_call_outcomes(
        self,
        period: str = "week",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get call outcome distribution.

        Args:
            period: Time period
            start_date: Start date for custom period
            end_date: End date for custom period
            agent_id: Filter by agent

        Returns:
            Call outcomes with counts and percentages
        """
        params = {"period": period}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if agent_id:
            params["agent_id"] = agent_id

        return self._get(f"{Endpoints.ANALYTICS_CALLS}/outcomes", params=params)

    def get_peak_hours(
        self,
        period: str = "week",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timezone: str = "UTC",
    ) -> Dict[str, Any]:
        """
        Get peak calling hours analysis.

        Args:
            period: Time period
            start_date: Start date for custom period
            end_date: End date for custom period
            timezone: Timezone for hour grouping

        Returns:
            Call volume by hour of day and day of week
        """
        params = {
            "period": period,
            "timezone": timezone,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return self._get(f"{Endpoints.ANALYTICS_CALLS}/peak-hours", params=params)

    def get_funnel(
        self,
        period: str = "week",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get call funnel analysis.

        Args:
            period: Time period
            start_date: Start date for custom period
            end_date: End date for custom period
            agent_id: Filter by agent

        Returns:
            Funnel data showing conversion at each stage
        """
        params = {"period": period}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if agent_id:
            params["agent_id"] = agent_id

        return self._get(f"{Endpoints.ANALYTICS_CALLS}/funnel", params=params)

    def compare_periods(
        self,
        current_period: str,
        previous_period: str,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare metrics between two periods.

        Args:
            current_period: Current time period
            previous_period: Previous time period to compare
            agent_id: Filter by agent

        Returns:
            Comparison data with percentage changes

        Example:
            >>> comparison = client.analytics.compare_periods(
            ...     current_period="this_week",
            ...     previous_period="last_week"
            ... )
            >>> print(f"Call volume change: {comparison['total_calls']['change']:.1%}")
        """
        params = {
            "current_period": current_period,
            "previous_period": previous_period,
        }
        if agent_id:
            params["agent_id"] = agent_id

        return self._get(f"{Endpoints.ANALYTICS_OVERVIEW}/compare", params=params)

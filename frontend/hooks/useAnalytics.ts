/**
 * Analytics Hook
 *
 * Provides analytics data fetching using TanStack Query.
 * Handles dashboard stats, call metrics, and agent performance.
 */

import { useQuery } from '@tanstack/react-query';
import { analyticsApi, AnalyticsParams, AnalyticsOverview, AnalyticsTimeSeries } from '@/lib/api';

// Query keys for cache management
export const analyticsKeys = {
  all: ['analytics'] as const,
  overview: (params: object) => [...analyticsKeys.all, 'overview', params] as const,
  timeSeries: (params: object) => [...analyticsKeys.all, 'timeSeries', params] as const,
  agentPerformance: (agentId: string) => [...analyticsKeys.all, 'agentPerformance', agentId] as const,
  callQuality: (params: object) => [...analyticsKeys.all, 'callQuality', params] as const,
  sentiment: (params: object) => [...analyticsKeys.all, 'sentiment', params] as const,
};

interface UseAnalyticsParams {
  fromDate?: string;
  toDate?: string;
  agentId?: string;
  granularity?: 'hour' | 'day' | 'week' | 'month';
}

/**
 * Hook for fetching analytics overview (summary)
 */
export function useAnalyticsSummary(params: UseAnalyticsParams = {}) {
  const apiParams: AnalyticsParams = {
    from_date: params.fromDate,
    to_date: params.toDate,
    agent_id: params.agentId,
  };

  return useQuery({
    queryKey: analyticsKeys.overview(params),
    queryFn: async () => {
      const response = await analyticsApi.overview(apiParams);
      return response.data;
    },
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for fetching time series data (calls by day/period)
 */
export function useCallsByDay(params: UseAnalyticsParams = {}) {
  const apiParams: AnalyticsParams = {
    from_date: params.fromDate,
    to_date: params.toDate,
    agent_id: params.agentId,
    granularity: params.granularity || 'day',
  };

  return useQuery({
    queryKey: analyticsKeys.timeSeries(params),
    queryFn: async () => {
      const response = await analyticsApi.timeSeries(apiParams);
      return response.data;
    },
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for fetching agent performance metrics
 */
export function useAgentPerformance(agentId: string, params: UseAnalyticsParams = {}) {
  const apiParams: AnalyticsParams = {
    from_date: params.fromDate,
    to_date: params.toDate,
  };

  return useQuery({
    queryKey: analyticsKeys.agentPerformance(agentId),
    queryFn: async () => {
      const response = await analyticsApi.agentPerformance(agentId, apiParams);
      return response.data;
    },
    enabled: !!agentId,
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for fetching dashboard stats (uses overview endpoint)
 */
export function useDashboardStats(params: UseAnalyticsParams = {}) {
  const apiParams: AnalyticsParams = {
    from_date: params.fromDate,
    to_date: params.toDate,
    agent_id: params.agentId,
  };

  return useQuery({
    queryKey: analyticsKeys.overview({ ...params, dashboard: true }),
    queryFn: async () => {
      const response = await analyticsApi.overview(apiParams);
      return response.data;
    },
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Auto-refresh every minute
  });
}

/**
 * Hook for fetching call quality metrics
 */
export function useCallQuality(params: UseAnalyticsParams = {}) {
  const apiParams: AnalyticsParams = {
    from_date: params.fromDate,
    to_date: params.toDate,
    agent_id: params.agentId,
  };

  return useQuery({
    queryKey: analyticsKeys.callQuality(params),
    queryFn: async () => {
      const response = await analyticsApi.callQuality(apiParams);
      return response.data;
    },
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for fetching sentiment analysis
 */
export function useSentimentAnalysis(params: UseAnalyticsParams = {}) {
  const apiParams: AnalyticsParams = {
    from_date: params.fromDate,
    to_date: params.toDate,
    agent_id: params.agentId,
  };

  return useQuery({
    queryKey: analyticsKeys.sentiment(params),
    queryFn: async () => {
      const response = await analyticsApi.sentimentAnalysis(apiParams);
      return response.data;
    },
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Helper hook for getting date range for common periods
 */
export function useDateRange(period: 'today' | 'week' | 'month' | 'year') {
  const now = new Date();
  const startOfDay = new Date(now.getFullYear(), now.getMonth(), now.getDate());

  let fromDate: Date;
  const toDate: Date = now;

  switch (period) {
    case 'today':
      fromDate = startOfDay;
      break;
    case 'week':
      fromDate = new Date(startOfDay);
      fromDate.setDate(fromDate.getDate() - 7);
      break;
    case 'month':
      fromDate = new Date(startOfDay);
      fromDate.setMonth(fromDate.getMonth() - 1);
      break;
    case 'year':
      fromDate = new Date(startOfDay);
      fromDate.setFullYear(fromDate.getFullYear() - 1);
      break;
    default:
      fromDate = startOfDay;
  }

  return {
    fromDate: fromDate.toISOString().split('T')[0],
    toDate: toDate.toISOString().split('T')[0],
  };
}

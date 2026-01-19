/**
 * Analytics Hook
 *
 * Provides analytics data fetching using TanStack Query.
 * Handles dashboard stats, call metrics, and agent performance.
 */

import { useQuery } from '@tanstack/react-query';
import { analyticsApi } from '@/lib/api';

// Query keys for cache management
export const analyticsKeys = {
  all: ['analytics'] as const,
  summary: (params: Record<string, any>) => [...analyticsKeys.all, 'summary', params] as const,
  callsByDay: (params: Record<string, any>) => [...analyticsKeys.all, 'callsByDay', params] as const,
  agentPerformance: () => [...analyticsKeys.all, 'agentPerformance'] as const,
  dashboard: () => [...analyticsKeys.all, 'dashboard'] as const,
};

interface AnalyticsParams {
  fromDate?: string;
  toDate?: string;
  agentId?: string;
}

/**
 * Hook for fetching analytics summary
 */
export function useAnalyticsSummary(params: AnalyticsParams = {}) {
  return useQuery({
    queryKey: analyticsKeys.summary(params),
    queryFn: () => analyticsApi.getSummary({
      from_date: params.fromDate,
      to_date: params.toDate,
      agent_id: params.agentId,
    }),
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for fetching calls by day chart data
 */
export function useCallsByDay(params: AnalyticsParams = {}) {
  return useQuery({
    queryKey: analyticsKeys.callsByDay(params),
    queryFn: () => analyticsApi.getCallsByDay({
      from_date: params.fromDate,
      to_date: params.toDate,
      agent_id: params.agentId,
    }),
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for fetching agent performance metrics
 */
export function useAgentPerformance() {
  return useQuery({
    queryKey: analyticsKeys.agentPerformance(),
    queryFn: () => analyticsApi.getAgentPerformance(),
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for fetching dashboard stats
 */
export function useDashboardStats() {
  return useQuery({
    queryKey: analyticsKeys.dashboard(),
    queryFn: () => analyticsApi.getDashboardStats(),
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Auto-refresh every minute
  });
}

/**
 * Helper hook for getting date range for common periods
 */
export function useDateRange(period: 'today' | 'week' | 'month' | 'year') {
  const now = new Date();
  const startOfDay = new Date(now.getFullYear(), now.getMonth(), now.getDate());

  let fromDate: Date;
  let toDate: Date = now;

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

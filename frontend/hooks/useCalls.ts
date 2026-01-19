/**
 * Calls Hook
 *
 * Provides call management and history using TanStack Query.
 * Handles call listing, details, and actions.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { callsApi } from '@/lib/api';
import { Call } from '@/types';

// Query keys for cache management
export const callKeys = {
  all: ['calls'] as const,
  lists: () => [...callKeys.all, 'list'] as const,
  list: (params: Record<string, any>) => [...callKeys.lists(), params] as const,
  details: () => [...callKeys.all, 'detail'] as const,
  detail: (id: string) => [...callKeys.details(), id] as const,
  conversation: (id: string) => [...callKeys.detail(id), 'conversation'] as const,
  events: (id: string) => [...callKeys.detail(id), 'events'] as const,
};

interface UseCallsParams {
  page?: number;
  pageSize?: number;
  agentId?: string;
  status?: string;
  direction?: string;
  fromDate?: string;
  toDate?: string;
}

/**
 * Hook for fetching paginated list of calls
 */
export function useCalls(params: UseCallsParams = {}) {
  return useQuery({
    queryKey: callKeys.list(params),
    queryFn: () => callsApi.list({
      page: params.page,
      page_size: params.pageSize,
      agent_id: params.agentId,
      status: params.status,
      direction: params.direction,
      from_date: params.fromDate,
      to_date: params.toDate,
    }),
    staleTime: 10 * 1000, // 10 seconds - calls update frequently
    refetchInterval: 30 * 1000, // Auto-refresh every 30 seconds
  });
}

/**
 * Hook for fetching a single call by ID
 */
export function useCall(id: string) {
  return useQuery({
    queryKey: callKeys.detail(id),
    queryFn: () => callsApi.get(id),
    enabled: !!id,
    refetchInterval: (query) => {
      // Auto-refresh active calls every 5 seconds
      const call = query.state.data as Call | undefined;
      if (call?.status === 'in_progress') {
        return 5 * 1000;
      }
      return false;
    },
  });
}

/**
 * Hook for fetching call conversation/transcript
 */
export function useCallConversation(callId: string) {
  return useQuery({
    queryKey: callKeys.conversation(callId),
    queryFn: () => callsApi.getConversation(callId),
    enabled: !!callId,
  });
}

/**
 * Hook for fetching call events
 */
export function useCallEvents(callId: string) {
  return useQuery({
    queryKey: callKeys.events(callId),
    queryFn: () => callsApi.getEvents(callId),
    enabled: !!callId,
  });
}

/**
 * Hook for initiating an outbound call
 */
export function useInitiateCall() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      agentId,
      toNumber,
      metadata,
    }: {
      agentId: string;
      toNumber: string;
      metadata?: Record<string, any>;
    }) => callsApi.initiateOutbound(agentId, toNumber, metadata),
    onSuccess: () => {
      // Invalidate call lists to show new call
      queryClient.invalidateQueries({ queryKey: callKeys.lists() });
    },
  });
}

/**
 * Hook for hanging up a call
 */
export function useHangupCall() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (callId: string) => callsApi.hangup(callId),
    onSuccess: (_, callId) => {
      // Invalidate specific call and lists
      queryClient.invalidateQueries({ queryKey: callKeys.detail(callId) });
      queryClient.invalidateQueries({ queryKey: callKeys.lists() });
    },
  });
}

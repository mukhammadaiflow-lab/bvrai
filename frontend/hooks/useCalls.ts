/**
 * Calls Hook
 *
 * Provides call management and history using TanStack Query.
 * Handles call listing, details, and actions.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { callsApi, Call, CallFilters, InitiateCallRequest } from '@/lib/api';

// Query keys for cache management
export const callKeys = {
  all: ['calls'] as const,
  lists: () => [...callKeys.all, 'list'] as const,
  list: (params: object) => [...callKeys.lists(), params] as const,
  details: () => [...callKeys.all, 'detail'] as const,
  detail: (id: string) => [...callKeys.details(), id] as const,
  transcript: (id: string) => [...callKeys.detail(id), 'transcript'] as const,
  recording: (id: string) => [...callKeys.detail(id), 'recording'] as const,
};

interface UseCallsParams {
  page?: number;
  limit?: number;
  agentId?: string;
  status?: Call['status'];
  direction?: Call['direction'];
  fromDate?: string;
  toDate?: string;
}

/**
 * Hook for fetching paginated list of calls
 */
export function useCalls(params: UseCallsParams = {}) {
  const apiParams: CallFilters = {
    page: params.page,
    limit: params.limit,
    agent_id: params.agentId,
    status: params.status,
    direction: params.direction,
    from_date: params.fromDate,
    to_date: params.toDate,
  };

  return useQuery({
    queryKey: callKeys.list(params),
    queryFn: async () => {
      const response = await callsApi.list(apiParams);
      return response.data;
    },
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
    queryFn: async () => {
      const response = await callsApi.get(id);
      return response.data;
    },
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
 * Hook for fetching call transcript
 */
export function useCallTranscript(callId: string) {
  return useQuery({
    queryKey: callKeys.transcript(callId),
    queryFn: async () => {
      const response = await callsApi.getTranscript(callId);
      return response.data;
    },
    enabled: !!callId,
  });
}

/**
 * Hook for fetching call recording URL
 */
export function useCallRecording(callId: string) {
  return useQuery({
    queryKey: callKeys.recording(callId),
    queryFn: async () => {
      const response = await callsApi.getRecording(callId);
      return response.data;
    },
    enabled: !!callId,
  });
}

/**
 * Hook for initiating an outbound call
 */
export function useInitiateCall() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: InitiateCallRequest) => {
      const response = await callsApi.initiate(data);
      return response.data;
    },
    onSuccess: () => {
      // Invalidate call lists to show new call
      queryClient.invalidateQueries({ queryKey: callKeys.lists() });
    },
  });
}

/**
 * Hook for ending a call
 */
export function useEndCall() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (callId: string) => {
      const response = await callsApi.end(callId);
      return response.data;
    },
    onSuccess: (_, callId) => {
      // Invalidate specific call and lists
      queryClient.invalidateQueries({ queryKey: callKeys.detail(callId) });
      queryClient.invalidateQueries({ queryKey: callKeys.lists() });
    },
  });
}

/**
 * Hook for adding a note to a call
 */
export function useAddCallNote() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ callId, note }: { callId: string; note: string }) => {
      const response = await callsApi.addNote(callId, note);
      return response.data;
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: callKeys.detail(variables.callId) });
    },
  });
}

// Legacy alias for backwards compatibility
export const useHangupCall = useEndCall;
export const useCallConversation = useCallTranscript;
export const useCallEvents = useCallTranscript;

"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { calls, Call } from "@/lib/api";
import { toast } from "sonner";

/**
 * Query keys for calls
 */
export const callKeys = {
  all: ["calls"] as const,
  lists: () => [...callKeys.all, "list"] as const,
  list: (filters: Record<string, unknown>) => [...callKeys.lists(), filters] as const,
  details: () => [...callKeys.all, "detail"] as const,
  detail: (id: string) => [...callKeys.details(), id] as const,
  analytics: () => [...callKeys.all, "analytics"] as const,
};

/**
 * Hook to fetch all calls
 */
export function useCalls(params?: {
  page?: number;
  limit?: number;
  status?: string;
  agentId?: string;
  startDate?: string;
  endDate?: string;
}) {
  return useQuery({
    queryKey: callKeys.list(params || {}),
    queryFn: () => calls.list(params),
  });
}

/**
 * Hook to fetch a single call by ID
 */
export function useCall(id: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: callKeys.detail(id),
    queryFn: () => calls.get(id),
    enabled: options?.enabled !== false && !!id,
  });
}

/**
 * Hook to initiate an outbound call
 */
export function useInitiateCall() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: { agentId: string; toNumber: string; metadata?: Record<string, unknown> }) =>
      calls.initiateOutbound(data.agentId, data.toNumber, data.metadata),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: callKeys.lists() });
      toast.success("Call initiated successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to initiate call: ${error.message}`);
    },
  });
}

/**
 * Hook to end an active call
 */
export function useEndCall() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => calls.end(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: callKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: callKeys.lists() });
      toast.success("Call ended");
    },
    onError: (error: Error) => {
      toast.error(`Failed to end call: ${error.message}`);
    },
  });
}

/**
 * Hook to fetch call analytics
 */
export function useCallAnalytics(params?: {
  startDate?: string;
  endDate?: string;
  agentId?: string;
}) {
  return useQuery({
    queryKey: [...callKeys.analytics(), params],
    queryFn: () => calls.getAnalytics(params),
    staleTime: 60 * 1000, // 1 minute stale time for analytics
  });
}

/**
 * Hook to get call transcript
 */
export function useCallTranscript(callId: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: [...callKeys.detail(callId), "transcript"],
    queryFn: () => calls.getTranscript(callId),
    enabled: options?.enabled !== false && !!callId,
  });
}

/**
 * Hook to get call recording URL
 */
export function useCallRecording(callId: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: [...callKeys.detail(callId), "recording"],
    queryFn: () => calls.getRecording(callId),
    enabled: options?.enabled !== false && !!callId,
    staleTime: 60 * 60 * 1000, // 1 hour stale time for recording URLs
  });
}

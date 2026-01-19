/**
 * Agents Hook
 *
 * Provides agent management using TanStack Query for server state.
 * Handles listing, creating, updating, and deleting agents.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { agentsApi } from '@/lib/api';
import { Agent, CreateAgentRequest, UpdateAgentRequest } from '@/types';

// Query keys for cache management
export const agentKeys = {
  all: ['agents'] as const,
  lists: () => [...agentKeys.all, 'list'] as const,
  list: (params: Record<string, any>) => [...agentKeys.lists(), params] as const,
  details: () => [...agentKeys.all, 'detail'] as const,
  detail: (id: string) => [...agentKeys.details(), id] as const,
  versions: (id: string) => [...agentKeys.detail(id), 'versions'] as const,
};

interface UseAgentsParams {
  page?: number;
  pageSize?: number;
  status?: string;
  search?: string;
}

/**
 * Hook for fetching paginated list of agents
 */
export function useAgents(params: UseAgentsParams = {}) {
  return useQuery({
    queryKey: agentKeys.list(params),
    queryFn: () => agentsApi.list({
      page: params.page,
      page_size: params.pageSize,
      status: params.status,
      search: params.search,
    }),
    staleTime: 30 * 1000, // 30 seconds
  });
}

/**
 * Hook for fetching a single agent by ID
 */
export function useAgent(id: string) {
  return useQuery({
    queryKey: agentKeys.detail(id),
    queryFn: () => agentsApi.get(id),
    enabled: !!id,
  });
}

/**
 * Hook for fetching agent versions
 */
export function useAgentVersions(id: string) {
  return useQuery({
    queryKey: agentKeys.versions(id),
    queryFn: () => agentsApi.getVersions(id),
    enabled: !!id,
  });
}

/**
 * Hook for creating a new agent
 */
export function useCreateAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateAgentRequest) => agentsApi.create(data),
    onSuccess: () => {
      // Invalidate all agent lists
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
    },
  });
}

/**
 * Hook for updating an agent
 */
export function useUpdateAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: UpdateAgentRequest }) =>
      agentsApi.update(id, data),
    onSuccess: (_, variables) => {
      // Invalidate specific agent and lists
      queryClient.invalidateQueries({ queryKey: agentKeys.detail(variables.id) });
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
    },
  });
}

/**
 * Hook for deleting an agent
 */
export function useDeleteAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => agentsApi.delete(id),
    onSuccess: (_, id) => {
      // Remove from cache and invalidate lists
      queryClient.removeQueries({ queryKey: agentKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
    },
  });
}

/**
 * Hook for duplicating an agent
 */
export function useDuplicateAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => agentsApi.duplicate(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
    },
  });
}

/**
 * Hook for rolling back to a previous agent version
 */
export function useRollbackAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, version }: { id: string; version: number }) =>
      agentsApi.rollback(id, version),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: agentKeys.detail(variables.id) });
      queryClient.invalidateQueries({ queryKey: agentKeys.versions(variables.id) });
    },
  });
}

/**
 * Hook for testing an agent with a message
 */
export function useTestAgent() {
  return useMutation({
    mutationFn: ({ id, message }: { id: string; message: string }) =>
      agentsApi.test(id, message),
  });
}

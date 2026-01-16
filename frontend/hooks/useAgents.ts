/**
 * Agents Hook
 *
 * Provides agent management using TanStack Query for server state.
 * Handles listing, creating, updating, and deleting agents.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { agentsApi, Agent, CreateAgentRequest, UpdateAgentRequest, ListParams } from '@/lib/api';

// Query keys for cache management
export const agentKeys = {
  all: ['agents'] as const,
  lists: () => [...agentKeys.all, 'list'] as const,
  list: (params: object) => [...agentKeys.lists(), params] as const,
  details: () => [...agentKeys.all, 'detail'] as const,
  detail: (id: string) => [...agentKeys.details(), id] as const,
  versions: (id: string) => [...agentKeys.detail(id), 'versions'] as const,
};

interface UseAgentsParams {
  page?: number;
  limit?: number;
  status?: string;
  search?: string;
}

/**
 * Hook for fetching paginated list of agents
 */
export function useAgents(params: UseAgentsParams = {}) {
  return useQuery({
    queryKey: agentKeys.list(params),
    queryFn: async () => {
      const response = await agentsApi.list({
        page: params.page,
        limit: params.limit,
        search: params.search,
      });
      return response.data;
    },
    staleTime: 30 * 1000, // 30 seconds
  });
}

/**
 * Hook for fetching a single agent by ID
 */
export function useAgent(id: string) {
  return useQuery({
    queryKey: agentKeys.detail(id),
    queryFn: async () => {
      const response = await agentsApi.get(id);
      return response.data;
    },
    enabled: !!id,
  });
}

/**
 * Hook for fetching agent versions
 */
export function useAgentVersions(id: string) {
  return useQuery({
    queryKey: agentKeys.versions(id),
    queryFn: async () => {
      const response = await agentsApi.getVersions(id);
      return response.data;
    },
    enabled: !!id,
  });
}

/**
 * Hook for creating a new agent
 */
export function useCreateAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: CreateAgentRequest) => {
      const response = await agentsApi.create(data);
      return response.data;
    },
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
    mutationFn: async ({ id, data }: { id: string; data: UpdateAgentRequest }) => {
      const response = await agentsApi.update(id, data);
      return response.data;
    },
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
    mutationFn: async (id: string) => {
      const response = await agentsApi.duplicate(id);
      return response.data;
    },
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
    mutationFn: async ({ id, version }: { id: string; version: number }) => {
      const response = await agentsApi.revertToVersion(id, version);
      return response.data;
    },
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

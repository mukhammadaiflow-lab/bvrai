"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { agents } from "@/lib/api";
import type { Agent, CreateAgentRequest, UpdateAgentRequest } from "@/types";
import { toast } from "sonner";

/**
 * Query keys for agents
 */
export const agentKeys = {
  all: ["agents"] as const,
  lists: () => [...agentKeys.all, "list"] as const,
  list: (filters: Record<string, unknown>) => [...agentKeys.lists(), filters] as const,
  details: () => [...agentKeys.all, "detail"] as const,
  detail: (id: string) => [...agentKeys.details(), id] as const,
};

/**
 * Hook to fetch all agents
 */
export function useAgents(params?: { page?: number; limit?: number; status?: string }) {
  return useQuery({
    queryKey: agentKeys.list(params || {}),
    queryFn: () => agents.list(params),
  });
}

/**
 * Hook to fetch a single agent by ID
 */
export function useAgent(id: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: agentKeys.detail(id),
    queryFn: () => agents.get(id),
    enabled: options?.enabled !== false && !!id,
  });
}

/**
 * Hook to create a new agent
 */
export function useCreateAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateAgentRequest) => agents.create(data),
    onSuccess: (newAgent) => {
      // Invalidate and refetch agents list
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
      toast.success("Agent created successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to create agent: ${error.message}`);
    },
  });
}

/**
 * Hook to update an agent
 */
export function useUpdateAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: UpdateAgentRequest }) => agents.update(id, data),
    onSuccess: (updatedAgent, { id }) => {
      // Update the cache directly
      queryClient.setQueryData(agentKeys.detail(id), updatedAgent);
      // Invalidate list to refetch
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
      toast.success("Agent updated successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to update agent: ${error.message}`);
    },
  });
}

/**
 * Hook to delete an agent
 */
export function useDeleteAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => agents.delete(id),
    onSuccess: (_, id) => {
      // Remove from cache
      queryClient.removeQueries({ queryKey: agentKeys.detail(id) });
      // Invalidate list
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
      toast.success("Agent deleted successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete agent: ${error.message}`);
    },
  });
}

/**
 * Hook to deploy an agent
 */
export function useDeployAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => agents.deploy(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: agentKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
      toast.success("Agent deployed successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to deploy agent: ${error.message}`);
    },
  });
}

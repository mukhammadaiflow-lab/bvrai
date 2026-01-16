/**
 * Organization Hook
 *
 * Provides organization and team management using TanStack Query.
 * Handles org settings, team members, and invitations.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  organizationApi,
  apiKeysApi,
  Organization,
  OrganizationMember,
  ApiKey,
  CreateApiKeyRequest
} from '@/lib/api';

// Query keys for cache management
export const organizationKeys = {
  all: ['organization'] as const,
  current: () => [...organizationKeys.all, 'current'] as const,
  members: () => [...organizationKeys.all, 'members'] as const,
};

export const apiKeyKeys = {
  all: ['apiKeys'] as const,
  lists: () => [...apiKeyKeys.all, 'list'] as const,
};

/**
 * Hook for fetching current organization
 */
export function useOrganization() {
  return useQuery({
    queryKey: organizationKeys.current(),
    queryFn: async () => {
      const response = await organizationApi.get();
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook for updating organization
 */
export function useUpdateOrganization() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: Partial<Organization>) => {
      const response = await organizationApi.update(data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: organizationKeys.current() });
    },
  });
}

/**
 * Hook for fetching team members
 */
export function useTeamMembers() {
  return useQuery({
    queryKey: organizationKeys.members(),
    queryFn: async () => {
      const response = await organizationApi.members();
      return response.data;
    },
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for inviting a team member
 */
export function useInviteMember() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ email, role }: { email: string; role: 'admin' | 'member' }) => {
      const response = await organizationApi.invite({ email, role });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: organizationKeys.members() });
    },
  });
}

/**
 * Hook for removing a team member
 */
export function useRemoveMember() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (memberId: string) => organizationApi.removeMember(memberId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: organizationKeys.members() });
    },
  });
}

/**
 * Hook for updating a member's role
 */
export function useUpdateMemberRole() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ memberId, role }: { memberId: string; role: OrganizationMember['role'] }) => {
      const response = await organizationApi.updateMemberRole(memberId, role);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: organizationKeys.members() });
    },
  });
}

/**
 * Hook for fetching API keys
 */
export function useApiKeys() {
  return useQuery({
    queryKey: apiKeyKeys.lists(),
    queryFn: async () => {
      const response = await apiKeysApi.list();
      return response.data;
    },
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for creating an API key
 */
export function useCreateApiKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: CreateApiKeyRequest) => {
      const response = await apiKeysApi.create(data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: apiKeyKeys.lists() });
    },
  });
}

/**
 * Hook for revoking an API key
 */
export function useRevokeApiKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => apiKeysApi.revoke(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: apiKeyKeys.lists() });
    },
  });
}

/**
 * Hook for regenerating an API key
 */
export function useRegenerateApiKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      const response = await apiKeysApi.regenerate(id);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: apiKeyKeys.lists() });
    },
  });
}

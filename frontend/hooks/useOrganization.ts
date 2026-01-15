/**
 * Organization Hook
 *
 * Provides organization and team management using TanStack Query.
 * Handles org settings, team members, and invitations.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { organizationApi, apiKeysApi } from '@/lib/api';
import { Organization, User, ApiKey } from '@/types';

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
    queryFn: () => organizationApi.getCurrent(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook for updating organization
 */
export function useUpdateOrganization() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: Partial<Organization>) => organizationApi.update(data),
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
    queryFn: () => organizationApi.getMembers(),
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for inviting a team member
 */
export function useInviteMember() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ email, role }: { email: string; role: string }) =>
      organizationApi.inviteMember(email, role),
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
    mutationFn: (userId: string) => organizationApi.removeMember(userId),
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
    queryFn: () => apiKeysApi.list(),
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for creating an API key
 */
export function useCreateApiKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name, scopes }: { name: string; scopes: string[] }) =>
      apiKeysApi.create(name, scopes),
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

/**
 * Webhooks Hook
 *
 * Provides webhook management using TanStack Query.
 * Handles CRUD operations and webhook testing.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { webhooksApi, Webhook, CreateWebhookRequest, WebhookDelivery, ListParams } from '@/lib/api';

// Query keys for cache management
export const webhookKeys = {
  all: ['webhooks'] as const,
  lists: () => [...webhookKeys.all, 'list'] as const,
  details: () => [...webhookKeys.all, 'detail'] as const,
  detail: (id: string) => [...webhookKeys.details(), id] as const,
  deliveries: (id: string) => [...webhookKeys.detail(id), 'deliveries'] as const,
  events: () => [...webhookKeys.all, 'events'] as const,
};

/**
 * Hook for fetching all webhooks
 */
export function useWebhooks() {
  return useQuery({
    queryKey: webhookKeys.lists(),
    queryFn: async () => {
      const response = await webhooksApi.list();
      return response.data;
    },
    staleTime: 30 * 1000, // 30 seconds
  });
}

/**
 * Hook for fetching a single webhook
 */
export function useWebhook(id: string) {
  return useQuery({
    queryKey: webhookKeys.detail(id),
    queryFn: async () => {
      const response = await webhooksApi.get(id);
      return response.data;
    },
    enabled: !!id,
  });
}

/**
 * Hook for fetching webhook deliveries
 */
export function useWebhookDeliveries(webhookId: string, params?: ListParams) {
  return useQuery({
    queryKey: webhookKeys.deliveries(webhookId),
    queryFn: async () => {
      const response = await webhooksApi.deliveries(webhookId, params);
      return response.data;
    },
    enabled: !!webhookId,
  });
}

/**
 * Hook for fetching available webhook events
 */
export function useWebhookEvents() {
  return useQuery({
    queryKey: webhookKeys.events(),
    queryFn: async () => {
      const response = await webhooksApi.listEvents();
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes - events don't change often
  });
}

/**
 * Hook for creating a webhook
 */
export function useCreateWebhook() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: CreateWebhookRequest) => {
      const response = await webhooksApi.create(data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: webhookKeys.lists() });
    },
  });
}

/**
 * Hook for updating a webhook
 */
export function useUpdateWebhook() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ id, data }: { id: string; data: Partial<CreateWebhookRequest> }) => {
      const response = await webhooksApi.update(id, data);
      return response.data;
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: webhookKeys.detail(variables.id) });
      queryClient.invalidateQueries({ queryKey: webhookKeys.lists() });
    },
  });
}

/**
 * Hook for deleting a webhook
 */
export function useDeleteWebhook() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => webhooksApi.delete(id),
    onSuccess: (_, id) => {
      queryClient.removeQueries({ queryKey: webhookKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: webhookKeys.lists() });
    },
  });
}

/**
 * Hook for testing a webhook
 */
export function useTestWebhook() {
  return useMutation({
    mutationFn: async (id: string) => {
      const response = await webhooksApi.test(id);
      return response.data;
    },
  });
}

/**
 * Hook for enabling a webhook
 */
export function useEnableWebhook() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => webhooksApi.enable(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: webhookKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: webhookKeys.lists() });
    },
  });
}

/**
 * Hook for disabling a webhook
 */
export function useDisableWebhook() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => webhooksApi.disable(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: webhookKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: webhookKeys.lists() });
    },
  });
}

/**
 * Hook for rotating webhook secret
 */
export function useRotateWebhookSecret() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      const response = await webhooksApi.rotateSecret(id);
      return response.data;
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: webhookKeys.detail(id) });
    },
  });
}

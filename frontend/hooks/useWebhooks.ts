/**
 * Webhooks Hook
 *
 * Provides webhook management using TanStack Query.
 * Handles CRUD operations and webhook testing.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { webhooksApi } from '@/lib/api';
import { Webhook, WebhookDelivery } from '@/types';

// Query keys for cache management
export const webhookKeys = {
  all: ['webhooks'] as const,
  lists: () => [...webhookKeys.all, 'list'] as const,
  details: () => [...webhookKeys.all, 'detail'] as const,
  detail: (id: string) => [...webhookKeys.details(), id] as const,
  deliveries: (id: string) => [...webhookKeys.detail(id), 'deliveries'] as const,
};

/**
 * Hook for fetching all webhooks
 */
export function useWebhooks() {
  return useQuery({
    queryKey: webhookKeys.lists(),
    queryFn: () => webhooksApi.list(),
    staleTime: 30 * 1000, // 30 seconds
  });
}

/**
 * Hook for fetching a single webhook
 */
export function useWebhook(id: string) {
  return useQuery({
    queryKey: webhookKeys.detail(id),
    queryFn: () => webhooksApi.get(id),
    enabled: !!id,
  });
}

/**
 * Hook for fetching webhook deliveries
 */
export function useWebhookDeliveries(webhookId: string) {
  return useQuery({
    queryKey: webhookKeys.deliveries(webhookId),
    queryFn: () => webhooksApi.getDeliveries(webhookId),
    enabled: !!webhookId,
  });
}

/**
 * Hook for creating a webhook
 */
export function useCreateWebhook() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: Partial<Webhook>) => webhooksApi.create(data),
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
    mutationFn: ({ id, data }: { id: string; data: Partial<Webhook> }) =>
      webhooksApi.update(id, data),
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
    mutationFn: (id: string) => webhooksApi.test(id),
  });
}

/**
 * Billing Hook
 *
 * Provides billing and subscription management using TanStack Query.
 * Handles plan info, usage, invoices, and subscription changes.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { billingApi, Subscription, UsageRecord, Invoice } from '@/lib/api';

// Query keys for cache management
export const billingKeys = {
  all: ['billing'] as const,
  subscription: () => [...billingKeys.all, 'subscription'] as const,
  invoices: () => [...billingKeys.all, 'invoices'] as const,
  usage: () => [...billingKeys.all, 'usage'] as const,
};

/**
 * Hook for fetching current subscription
 */
export function useSubscription() {
  return useQuery({
    queryKey: billingKeys.subscription(),
    queryFn: async () => {
      const response = await billingApi.subscription();
      return response.data;
    },
    staleTime: 60 * 1000, // 1 minute
  });
}

// Alias for backwards compatibility
export const useCurrentPlan = useSubscription;

/**
 * Hook for fetching invoices
 */
export function useInvoices() {
  return useQuery({
    queryKey: billingKeys.invoices(),
    queryFn: async () => {
      const response = await billingApi.invoices();
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook for fetching usage data
 */
export function useUsage() {
  return useQuery({
    queryKey: billingKeys.usage(),
    queryFn: async () => {
      const response = await billingApi.usage();
      return response.data;
    },
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for creating a checkout session (upgrading plan)
 */
export function useCreateCheckout() {
  return useMutation({
    mutationFn: async (priceId: string) => {
      const response = await billingApi.createCheckout(priceId);
      return response.data;
    },
    onSuccess: (data) => {
      // Redirect to checkout
      if (data && data.checkout_url) {
        window.location.href = data.checkout_url;
      }
    },
  });
}

/**
 * Hook for cancelling subscription
 */
export function useCancelSubscription() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      const response = await billingApi.cancelSubscription();
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: billingKeys.subscription() });
    },
  });
}

/**
 * Hook for resuming a cancelled subscription
 */
export function useResumeSubscription() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      const response = await billingApi.resumeSubscription();
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: billingKeys.subscription() });
    },
  });
}

/**
 * Hook for opening billing portal
 */
export function useCreatePortalSession() {
  return useMutation({
    mutationFn: async () => {
      const response = await billingApi.createPortalSession();
      return response.data;
    },
    onSuccess: (data) => {
      if (data && data.portal_url) {
        window.location.href = data.portal_url;
      }
    },
  });
}

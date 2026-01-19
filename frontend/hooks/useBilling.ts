/**
 * Billing Hook
 *
 * Provides billing and subscription management using TanStack Query.
 * Handles plan info, usage, invoices, and subscription changes.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { billingApi } from '@/lib/api';
import { Invoice, UsageSummary } from '@/types';

// Query keys for cache management
export const billingKeys = {
  all: ['billing'] as const,
  plan: () => [...billingKeys.all, 'plan'] as const,
  invoices: () => [...billingKeys.all, 'invoices'] as const,
  usage: (params?: { from_date?: string; to_date?: string }) =>
    [...billingKeys.all, 'usage', params] as const,
};

/**
 * Hook for fetching current plan and usage
 */
export function useCurrentPlan() {
  return useQuery({
    queryKey: billingKeys.plan(),
    queryFn: () => billingApi.getCurrentPlan(),
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for fetching invoices
 */
export function useInvoices() {
  return useQuery({
    queryKey: billingKeys.invoices(),
    queryFn: () => billingApi.getInvoices(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook for fetching usage data
 */
export function useUsage(params?: { from_date?: string; to_date?: string }) {
  return useQuery({
    queryKey: billingKeys.usage(params),
    queryFn: () => billingApi.getUsage(params),
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for creating a checkout session (upgrading plan)
 */
export function useCreateCheckout() {
  return useMutation({
    mutationFn: (planId: string) => billingApi.createCheckoutSession(planId),
    onSuccess: (data) => {
      // Redirect to checkout
      if (data.checkout_url) {
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
    mutationFn: () => billingApi.cancelSubscription(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: billingKeys.plan() });
    },
  });
}

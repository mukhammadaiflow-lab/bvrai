"use client";

import { useState, useCallback, useRef } from "react";

interface UseFormSubmitOptions {
  resetOnSuccess?: boolean;
}

interface UseFormSubmitReturn<T> {
  handleSubmit: (data: T) => Promise<void>;
  isSubmitting: boolean;
  error: Error | null;
  clearError: () => void;
}

/**
 * Custom hook for handling form submissions with double-submit prevention
 * and automatic loading state management.
 *
 * @example
 * const { handleSubmit, isSubmitting, error } = useFormSubmit<FormData>(async (data) => {
 *   await api.updateProfile(data);
 * });
 *
 * <form onSubmit={(e) => { e.preventDefault(); handleSubmit(formData); }}>
 *   <Button disabled={isSubmitting}>
 *     {isSubmitting ? "Saving..." : "Save"}
 *   </Button>
 * </form>
 */
export function useFormSubmit<T>(
  onSubmit: (data: T) => Promise<void>,
  options: UseFormSubmitOptions = {}
): UseFormSubmitReturn<T> {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const isSubmittingRef = useRef(false);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const handleSubmit = useCallback(
    async (data: T) => {
      // Prevent double submission using both state and ref
      // Ref prevents race conditions, state triggers re-renders
      if (isSubmittingRef.current) {
        return;
      }

      isSubmittingRef.current = true;
      setIsSubmitting(true);
      setError(null);

      try {
        await onSubmit(data);
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        throw error;
      } finally {
        isSubmittingRef.current = false;
        setIsSubmitting(false);
      }
    },
    [onSubmit]
  );

  return {
    handleSubmit,
    isSubmitting,
    error,
    clearError,
  };
}

/**
 * Simple hook for button click handlers with double-click prevention
 */
export function useSubmitButton(onClick: () => Promise<void>) {
  const [isLoading, setIsLoading] = useState(false);
  const isLoadingRef = useRef(false);

  const handleClick = useCallback(async () => {
    if (isLoadingRef.current) return;

    isLoadingRef.current = true;
    setIsLoading(true);

    try {
      await onClick();
    } finally {
      isLoadingRef.current = false;
      setIsLoading(false);
    }
  }, [onClick]);

  return { handleClick, isLoading };
}

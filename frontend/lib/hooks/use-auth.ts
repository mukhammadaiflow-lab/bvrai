"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { auth } from "@/lib/api";
import type { User } from "@/types";
import { toast } from "sonner";
import { useRouter } from "next/navigation";

/**
 * Query keys for auth
 */
export const authKeys = {
  user: ["auth", "user"] as const,
  session: ["auth", "session"] as const,
};

// Token storage helpers
const TOKEN_KEY = "bvrai_access_token";

export function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setStoredToken(token: string): void {
  if (typeof window !== "undefined") {
    localStorage.setItem(TOKEN_KEY, token);
  }
}

export function clearStoredToken(): void {
  if (typeof window !== "undefined") {
    localStorage.removeItem(TOKEN_KEY);
  }
}

/**
 * Hook to get current user
 */
export function useCurrentUser() {
  return useQuery({
    queryKey: authKeys.user,
    queryFn: async () => {
      const token = getStoredToken();
      if (!token) {
        throw new Error("No auth token");
      }
      return auth.getCurrentUser();
    },
    retry: false,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook to login
 */
export function useLogin() {
  const queryClient = useQueryClient();
  const router = useRouter();

  return useMutation({
    mutationFn: (credentials: { email: string; password: string }) =>
      auth.login(credentials.email, credentials.password),
    onSuccess: (data) => {
      // Store the token
      const token = data.access_token || data.token;
      if (token) {
        setStoredToken(token);
      }

      // Update user in cache
      queryClient.setQueryData(authKeys.user, data.user);

      toast.success("Logged in successfully");
      router.push("/dashboard");
    },
    onError: (error: Error) => {
      toast.error(`Login failed: ${error.message}`);
    },
  });
}

/**
 * Hook to register
 */
export function useRegister() {
  const queryClient = useQueryClient();
  const router = useRouter();

  return useMutation({
    mutationFn: (data: {
      email: string;
      password: string;
      name?: string;
      organizationName?: string;
    }) => auth.register(data.email, data.password, data.name, data.organizationName),
    onSuccess: (data) => {
      // Store the token
      if (data.access_token) {
        setStoredToken(data.access_token);
      }

      // Update user in cache
      queryClient.setQueryData(authKeys.user, data.user);

      toast.success("Account created successfully");
      router.push("/dashboard");
    },
    onError: (error: Error) => {
      toast.error(`Registration failed: ${error.message}`);
    },
  });
}

/**
 * Hook to logout
 */
export function useLogout() {
  const queryClient = useQueryClient();
  const router = useRouter();

  return useMutation({
    mutationFn: () => auth.logout(),
    onSettled: () => {
      // Always clear on logout attempt
      clearStoredToken();
      queryClient.clear();
      router.push("/auth/login");
      toast.success("Logged out successfully");
    },
  });
}

/**
 * Hook to check if user is authenticated
 */
export function useIsAuthenticated() {
  const { data: user, isLoading, isError } = useCurrentUser();

  return {
    isAuthenticated: !!user && !isError,
    isLoading,
    user,
  };
}

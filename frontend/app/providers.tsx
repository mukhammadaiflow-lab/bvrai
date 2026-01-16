"use client";

import { useState, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { Toaster } from "sonner";

/**
 * Application Providers
 *
 * Wraps the application with necessary context providers:
 * - React Query for server state management
 * - Toast notifications
 * - Error boundaries
 */
export function Providers({ children }: { children: ReactNode }) {
  // Create QueryClient instance once per app lifecycle
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Stale time: how long data is considered fresh
            staleTime: 60 * 1000, // 1 minute
            // Cache time: how long to keep unused data
            gcTime: 5 * 60 * 1000, // 5 minutes (formerly cacheTime)
            // Retry failed requests
            retry: (failureCount, error: unknown) => {
              // Don't retry on 4xx errors
              if (error && typeof error === "object" && "status" in error) {
                const status = (error as { status: number }).status;
                if (status >= 400 && status < 500) {
                  return false;
                }
              }
              return failureCount < 3;
            },
            // Retry delay with exponential backoff
            retryDelay: (attemptIndex) =>
              Math.min(1000 * 2 ** attemptIndex, 30000),
            // Refetch on window focus (good for data freshness)
            refetchOnWindowFocus: true,
            // Don't refetch on reconnect by default
            refetchOnReconnect: true,
          },
          mutations: {
            // Retry mutations once
            retry: 1,
            // Network error retry
            networkMode: "always",
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <Toaster
        position="top-right"
        expand={false}
        richColors
        closeButton
        duration={5000}
        toastOptions={{
          style: {
            background: "hsl(var(--background))",
            color: "hsl(var(--foreground))",
            border: "1px solid hsl(var(--border))",
          },
        }}
      />
      {/* React Query Devtools - only in development */}
      {process.env.NODE_ENV === "development" && (
        <ReactQueryDevtools initialIsOpen={false} buttonPosition="bottom-left" />
      )}
    </QueryClientProvider>
  );
}

/**
 * Error Boundary Component
 * Catches JavaScript errors in child components
 */
import React from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log error to monitoring service
    console.error("Error caught by boundary:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen flex items-center justify-center bg-background">
          <div className="text-center p-8 max-w-md">
            <h2 className="text-2xl font-bold text-foreground mb-4">
              Something went wrong
            </h2>
            <p className="text-muted-foreground mb-6">
              An unexpected error occurred. Please try refreshing the page.
            </p>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: undefined });
                window.location.reload();
              }}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
            >
              Refresh Page
            </button>
            {process.env.NODE_ENV === "development" && this.state.error && (
              <pre className="mt-6 p-4 bg-destructive/10 text-destructive text-left text-sm rounded-md overflow-auto max-h-48">
                {this.state.error.message}
                {"\n"}
                {this.state.error.stack}
              </pre>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Auth Provider Wrapper
 * Handles hydration of auth state from localStorage
 */
export function AuthHydration({ children }: { children: ReactNode }) {
  const [isHydrated, setIsHydrated] = useState(false);

  // Hydrate auth state after mount to avoid SSR mismatch
  useState(() => {
    setIsHydrated(true);
  });

  // Show nothing during hydration to prevent flash
  if (!isHydrated) {
    return null;
  }

  return <>{children}</>;
}

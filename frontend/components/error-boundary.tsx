"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RefreshCw, Home, Bug } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  showDetails?: boolean;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * Error Boundary Component
 *
 * Catches JavaScript errors anywhere in the child component tree,
 * logs those errors, and displays a fallback UI.
 *
 * Usage:
 * <ErrorBoundary>
 *   <YourComponent />
 * </ErrorBoundary>
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });

    // Log error to console in development
    if (process.env.NODE_ENV === "development") {
      console.error("Error Boundary caught an error:", error, errorInfo);
    }

    // Call optional error handler
    this.props.onError?.(error, errorInfo);

    // Send to error tracking service (e.g., Sentry)
    this.logErrorToService(error, errorInfo);
  }

  private logErrorToService(error: Error, errorInfo: ErrorInfo): void {
    // TODO: Integrate with error tracking service
    // Example: Sentry.captureException(error, { extra: { componentStack: errorInfo.componentStack } });

    // For now, log to console
    console.error("[Error Boundary]", {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
    });
  }

  private handleRetry = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  private handleGoHome = (): void => {
    window.location.href = "/dashboard";
  };

  private handleReportBug = (): void => {
    const error = this.state.error;
    const errorInfo = this.state.errorInfo;

    // Create bug report URL or open modal
    const errorDetails = encodeURIComponent(
      `Error: ${error?.message}\n\nStack: ${error?.stack}\n\nComponent Stack: ${errorInfo?.componentStack}`
    );

    // In production, this would open a bug report form
    console.log("Bug report data:", errorDetails);
    alert("Bug report feature - would open support form in production");
  };

  render(): ReactNode {
    const { hasError, error, errorInfo } = this.state;
    const { children, fallback, showDetails = false } = this.props;

    if (hasError) {
      // Custom fallback provided
      if (fallback) {
        return fallback;
      }

      // Default error UI
      return (
        <div className="flex min-h-[400px] items-center justify-center p-6">
          <Card className="w-full max-w-lg border-destructive/50">
            <CardHeader className="text-center">
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-destructive/10">
                <AlertTriangle className="h-8 w-8 text-destructive" />
              </div>
              <CardTitle className="text-xl">Something went wrong</CardTitle>
              <CardDescription>
                We encountered an unexpected error. Please try again or contact support if the problem persists.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Error message in development */}
              {(showDetails || process.env.NODE_ENV === "development") && error && (
                <div className="rounded-md bg-muted p-4">
                  <p className="mb-2 text-sm font-medium text-muted-foreground">Error Details:</p>
                  <pre className="overflow-auto text-xs text-destructive">
                    {error.message}
                  </pre>
                  {errorInfo?.componentStack && (
                    <>
                      <p className="mb-2 mt-4 text-sm font-medium text-muted-foreground">Component Stack:</p>
                      <pre className="max-h-32 overflow-auto text-xs text-muted-foreground">
                        {errorInfo.componentStack}
                      </pre>
                    </>
                  )}
                </div>
              )}

              {/* Action buttons */}
              <div className="flex flex-col gap-2 sm:flex-row">
                <Button onClick={this.handleRetry} className="flex-1">
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Try Again
                </Button>
                <Button variant="outline" onClick={this.handleGoHome} className="flex-1">
                  <Home className="mr-2 h-4 w-4" />
                  Go to Dashboard
                </Button>
              </div>

              <Button
                variant="ghost"
                onClick={this.handleReportBug}
                className="w-full text-muted-foreground"
              >
                <Bug className="mr-2 h-4 w-4" />
                Report this issue
              </Button>
            </CardContent>
          </Card>
        </div>
      );
    }

    return children;
  }
}

/**
 * Hook version for functional components
 * Note: React doesn't have a hook equivalent to componentDidCatch,
 * but this provides a pattern for error state management
 */
interface UseErrorBoundaryResult {
  hasError: boolean;
  error: Error | null;
  resetError: () => void;
  captureError: (error: Error) => void;
}

export function useErrorBoundary(): UseErrorBoundaryResult {
  const [error, setError] = React.useState<Error | null>(null);

  const resetError = React.useCallback(() => {
    setError(null);
  }, []);

  const captureError = React.useCallback((error: Error) => {
    setError(error);
  }, []);

  return {
    hasError: error !== null,
    error,
    resetError,
    captureError,
  };
}

/**
 * Higher-order component for wrapping components with error boundary
 */
export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ErrorBoundaryProps, "children">
): React.FC<P> {
  const displayName = WrappedComponent.displayName || WrappedComponent.name || "Component";

  const ComponentWithErrorBoundary: React.FC<P> = (props) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <WrappedComponent {...props} />
    </ErrorBoundary>
  );

  ComponentWithErrorBoundary.displayName = `withErrorBoundary(${displayName})`;

  return ComponentWithErrorBoundary;
}

export default ErrorBoundary;

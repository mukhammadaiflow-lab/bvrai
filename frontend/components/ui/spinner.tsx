"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface SpinnerProps extends React.SVGAttributes<SVGSVGElement> {
  size?: "xs" | "sm" | "default" | "lg" | "xl";
  variant?: "default" | "primary" | "secondary" | "muted";
}

const Spinner = React.forwardRef<SVGSVGElement, SpinnerProps>(
  ({ className, size = "default", variant = "default", ...props }, ref) => {
    const sizeClasses = {
      xs: "h-3 w-3",
      sm: "h-4 w-4",
      default: "h-6 w-6",
      lg: "h-8 w-8",
      xl: "h-12 w-12",
    };

    const variantClasses = {
      default: "text-foreground",
      primary: "text-primary",
      secondary: "text-secondary-foreground",
      muted: "text-muted-foreground",
    };

    return (
      <svg
        ref={ref}
        className={cn(
          "animate-spin",
          sizeClasses[size],
          variantClasses[variant],
          className
        )}
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        {...props}
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
    );
  }
);
Spinner.displayName = "Spinner";

interface LoadingOverlayProps extends React.HTMLAttributes<HTMLDivElement> {
  text?: string;
  size?: "xs" | "sm" | "default" | "lg" | "xl";
}

const LoadingOverlay = React.forwardRef<HTMLDivElement, LoadingOverlayProps>(
  ({ className, text, size = "lg", ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "absolute inset-0 flex flex-col items-center justify-center bg-background/80 backdrop-blur-sm z-50",
        className
      )}
      {...props}
    >
      <Spinner size={size} variant="primary" />
      {text && (
        <p className="mt-3 text-sm text-muted-foreground animate-pulse">
          {text}
        </p>
      )}
    </div>
  )
);
LoadingOverlay.displayName = "LoadingOverlay";

interface DotsLoaderProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: "sm" | "default" | "lg";
}

const DotsLoader = React.forwardRef<HTMLDivElement, DotsLoaderProps>(
  ({ className, size = "default", ...props }, ref) => {
    const sizeClasses = {
      sm: "h-1.5 w-1.5",
      default: "h-2 w-2",
      lg: "h-3 w-3",
    };

    return (
      <div
        ref={ref}
        className={cn("flex items-center space-x-1", className)}
        {...props}
      >
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className={cn(
              "rounded-full bg-current animate-bounce",
              sizeClasses[size]
            )}
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
    );
  }
);
DotsLoader.displayName = "DotsLoader";

interface PulseLoaderProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: "sm" | "default" | "lg";
}

const PulseLoader = React.forwardRef<HTMLDivElement, PulseLoaderProps>(
  ({ className, size = "default", ...props }, ref) => {
    const sizeClasses = {
      sm: "h-8 w-8",
      default: "h-12 w-12",
      lg: "h-16 w-16",
    };

    return (
      <div
        ref={ref}
        className={cn("relative", sizeClasses[size], className)}
        {...props}
      >
        <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping" />
        <div className="absolute inset-2 rounded-full bg-primary/40 animate-pulse" />
        <div className="absolute inset-4 rounded-full bg-primary" />
      </div>
    );
  }
);
PulseLoader.displayName = "PulseLoader";

export { Spinner, LoadingOverlay, DotsLoader, PulseLoader };

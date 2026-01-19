"use client";

import * as React from "react";
import * as ProgressPrimitive from "@radix-ui/react-progress";
import { cn } from "@/lib/utils";

interface ProgressProps
  extends React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root> {
  indicatorClassName?: string;
  showLabel?: boolean;
  size?: "sm" | "default" | "lg";
  variant?: "default" | "success" | "warning" | "destructive";
}

const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  ProgressProps
>(
  (
    {
      className,
      value,
      indicatorClassName,
      showLabel,
      size = "default",
      variant = "default",
      ...props
    },
    ref
  ) => {
    const sizeClasses = {
      sm: "h-1.5",
      default: "h-2.5",
      lg: "h-4",
    };

    const variantClasses = {
      default: "bg-primary",
      success: "bg-green-500",
      warning: "bg-yellow-500",
      destructive: "bg-destructive",
    };

    return (
      <div className="w-full">
        <ProgressPrimitive.Root
          ref={ref}
          className={cn(
            "relative w-full overflow-hidden rounded-full bg-secondary",
            sizeClasses[size],
            className
          )}
          {...props}
        >
          <ProgressPrimitive.Indicator
            className={cn(
              "h-full w-full flex-1 transition-all duration-300 ease-in-out",
              variantClasses[variant],
              indicatorClassName
            )}
            style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
          />
        </ProgressPrimitive.Root>
        {showLabel && (
          <div className="mt-1 flex justify-between text-xs text-muted-foreground">
            <span>{value}%</span>
            <span>100%</span>
          </div>
        )}
      </div>
    );
  }
);
Progress.displayName = ProgressPrimitive.Root.displayName;

interface CircularProgressProps {
  value: number;
  size?: number;
  strokeWidth?: number;
  className?: string;
  showLabel?: boolean;
  variant?: "default" | "success" | "warning" | "destructive";
}

const CircularProgress = React.forwardRef<SVGSVGElement, CircularProgressProps>(
  (
    {
      value,
      size = 60,
      strokeWidth = 6,
      className,
      showLabel = true,
      variant = "default",
    },
    ref
  ) => {
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (value / 100) * circumference;

    const variantColors = {
      default: "stroke-primary",
      success: "stroke-green-500",
      warning: "stroke-yellow-500",
      destructive: "stroke-destructive",
    };

    return (
      <div className={cn("relative inline-flex items-center justify-center", className)}>
        <svg
          ref={ref}
          width={size}
          height={size}
          className="transform -rotate-90"
        >
          <circle
            className="stroke-secondary"
            fill="transparent"
            strokeWidth={strokeWidth}
            r={radius}
            cx={size / 2}
            cy={size / 2}
          />
          <circle
            className={cn(
              "transition-all duration-300 ease-in-out",
              variantColors[variant]
            )}
            fill="transparent"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            r={radius}
            cx={size / 2}
            cy={size / 2}
          />
        </svg>
        {showLabel && (
          <span className="absolute text-sm font-semibold">
            {Math.round(value)}%
          </span>
        )}
      </div>
    );
  }
);
CircularProgress.displayName = "CircularProgress";

export { Progress, CircularProgress };

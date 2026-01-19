"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { Button } from "./button";
import {
  FileQuestion,
  Search,
  Inbox,
  FolderOpen,
  AlertCircle,
  Wifi,
  Server,
} from "lucide-react";

type EmptyStateVariant =
  | "no-data"
  | "no-results"
  | "empty-inbox"
  | "empty-folder"
  | "error"
  | "offline"
  | "custom";

interface EmptyStateProps {
  variant?: EmptyStateVariant;
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
    variant?: "default" | "outline" | "secondary";
  };
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
  size?: "sm" | "default" | "lg";
}

const variantIcons: Record<EmptyStateVariant, React.ReactNode> = {
  "no-data": <FileQuestion className="h-12 w-12" />,
  "no-results": <Search className="h-12 w-12" />,
  "empty-inbox": <Inbox className="h-12 w-12" />,
  "empty-folder": <FolderOpen className="h-12 w-12" />,
  error: <AlertCircle className="h-12 w-12" />,
  offline: <Wifi className="h-12 w-12" />,
  custom: <Server className="h-12 w-12" />,
};

function EmptyState({
  variant = "no-data",
  icon,
  title,
  description,
  action,
  secondaryAction,
  className,
  size = "default",
}: EmptyStateProps) {
  const sizeClasses = {
    sm: "py-8",
    default: "py-12",
    lg: "py-16",
  };

  const iconSizeClasses = {
    sm: "[&_svg]:h-8 [&_svg]:w-8",
    default: "[&_svg]:h-12 [&_svg]:w-12",
    lg: "[&_svg]:h-16 [&_svg]:w-16",
  };

  const titleSizeClasses = {
    sm: "text-base",
    default: "text-lg",
    lg: "text-xl",
  };

  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center text-center px-4",
        sizeClasses[size],
        className
      )}
    >
      <div
        className={cn(
          "rounded-full bg-muted p-4 text-muted-foreground mb-4",
          iconSizeClasses[size]
        )}
      >
        {icon || variantIcons[variant]}
      </div>
      <h3 className={cn("font-semibold mb-1", titleSizeClasses[size])}>
        {title}
      </h3>
      {description && (
        <p className="text-sm text-muted-foreground max-w-sm mb-4">
          {description}
        </p>
      )}
      {(action || secondaryAction) && (
        <div className="flex items-center gap-2 mt-2">
          {action && (
            <Button
              variant={action.variant || "default"}
              onClick={action.onClick}
              size={size === "sm" ? "sm" : "default"}
            >
              {action.label}
            </Button>
          )}
          {secondaryAction && (
            <Button
              variant="ghost"
              onClick={secondaryAction.onClick}
              size={size === "sm" ? "sm" : "default"}
            >
              {secondaryAction.label}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}

interface NoDataProps {
  title?: string;
  description?: string;
  action?: EmptyStateProps["action"];
  className?: string;
}

function NoData({
  title = "No data available",
  description = "There's nothing here yet. Start by creating something new.",
  action,
  className,
}: NoDataProps) {
  return (
    <EmptyState
      variant="no-data"
      title={title}
      description={description}
      action={action}
      className={className}
    />
  );
}

function NoSearchResults({
  query,
  onClearSearch,
  className,
}: {
  query?: string;
  onClearSearch?: () => void;
  className?: string;
}) {
  return (
    <EmptyState
      variant="no-results"
      title="No results found"
      description={
        query
          ? `No results found for "${query}". Try adjusting your search or filters.`
          : "Try adjusting your search or filters to find what you're looking for."
      }
      action={
        onClearSearch
          ? { label: "Clear search", onClick: onClearSearch, variant: "outline" }
          : undefined
      }
      className={className}
    />
  );
}

function ErrorState({
  title = "Something went wrong",
  description = "An error occurred while loading this content. Please try again.",
  onRetry,
  className,
}: {
  title?: string;
  description?: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <EmptyState
      variant="error"
      title={title}
      description={description}
      action={onRetry ? { label: "Try again", onClick: onRetry } : undefined}
      className={className}
    />
  );
}

function OfflineState({
  onRetry,
  className,
}: {
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <EmptyState
      variant="offline"
      title="You're offline"
      description="Please check your internet connection and try again."
      action={onRetry ? { label: "Retry", onClick: onRetry } : undefined}
      className={className}
    />
  );
}

export {
  EmptyState,
  NoData,
  NoSearchResults,
  ErrorState,
  OfflineState,
};

"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { Card, CardContent } from "./card";
import { TrendingUp, TrendingDown, Minus, ArrowRight } from "lucide-react";
import { Skeleton } from "./skeleton";

interface StatCardProps {
  title: string;
  value: string | number;
  description?: string;
  icon?: React.ReactNode;
  trend?: {
    value: number;
    label?: string;
    direction?: "up" | "down" | "neutral";
  };
  className?: string;
  loading?: boolean;
  onClick?: () => void;
  href?: string;
}

function StatCard({
  title,
  value,
  description,
  icon,
  trend,
  className,
  loading,
  onClick,
  href,
}: StatCardProps) {
  const trendDirection = trend?.direction || (trend?.value && trend.value > 0 ? "up" : trend?.value && trend.value < 0 ? "down" : "neutral");

  const trendColors = {
    up: "text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-950",
    down: "text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-950",
    neutral: "text-muted-foreground bg-muted",
  };

  const TrendIcon = trendDirection === "up" ? TrendingUp : trendDirection === "down" ? TrendingDown : Minus;

  const content = (
    <CardContent className="p-6">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          {loading ? (
            <Skeleton className="h-8 w-24 mt-2" />
          ) : (
            <p className="text-2xl font-bold mt-1">{value}</p>
          )}
          {description && (
            <p className="text-xs text-muted-foreground mt-1">{description}</p>
          )}
        </div>
        {icon && (
          <div className="rounded-lg bg-primary/10 p-3 text-primary">
            {icon}
          </div>
        )}
      </div>
      {trend && !loading && (
        <div className="mt-4 flex items-center gap-2">
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium",
              trendColors[trendDirection]
            )}
          >
            <TrendIcon className="h-3 w-3" />
            {Math.abs(trend.value)}%
          </span>
          {trend.label && (
            <span className="text-xs text-muted-foreground">{trend.label}</span>
          )}
        </div>
      )}
      {(onClick || href) && (
        <div className="mt-4 flex items-center text-sm text-primary">
          <span>View details</span>
          <ArrowRight className="ml-1 h-4 w-4" />
        </div>
      )}
    </CardContent>
  );

  const cardClass = cn(
    "transition-all duration-200",
    (onClick || href) && "cursor-pointer hover:shadow-md hover:border-primary/50",
    className
  );

  if (href) {
    return (
      <a href={href} className="block">
        <Card className={cardClass}>
          {content}
        </Card>
      </a>
    );
  }

  return (
    <Card className={cardClass} onClick={onClick}>
      {content}
    </Card>
  );
}

interface StatGridProps {
  children: React.ReactNode;
  columns?: 2 | 3 | 4;
  className?: string;
}

function StatGrid({ children, columns = 4, className }: StatGridProps) {
  const columnClasses = {
    2: "grid-cols-1 md:grid-cols-2",
    3: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 md:grid-cols-2 lg:grid-cols-4",
  };

  return (
    <div className={cn("grid gap-4", columnClasses[columns], className)}>
      {children}
    </div>
  );
}

interface MiniStatProps {
  label: string;
  value: string | number;
  change?: number;
  className?: string;
}

function MiniStat({ label, value, change, className }: MiniStatProps) {
  return (
    <div className={cn("flex items-center justify-between py-2", className)}>
      <span className="text-sm text-muted-foreground">{label}</span>
      <div className="flex items-center gap-2">
        <span className="font-medium">{value}</span>
        {change !== undefined && (
          <span
            className={cn(
              "text-xs",
              change > 0 ? "text-green-600" : change < 0 ? "text-red-600" : "text-muted-foreground"
            )}
          >
            {change > 0 ? "+" : ""}{change}%
          </span>
        )}
      </div>
    </div>
  );
}

export { StatCard, StatGrid, MiniStat };

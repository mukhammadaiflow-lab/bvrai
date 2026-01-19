"use client";

import React from "react";
import Link from "next/link";
import { Phone, Clock, CreditCard, AlertTriangle, TrendingUp, ArrowRight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress, CircularProgress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn, formatNumber } from "@/lib/utils";

interface UsageData {
  calls_used: number;
  calls_limit: number;
  minutes_used: number;
  minutes_limit: number;
  billing_period_end?: string;
}

interface UsageStatsProps {
  usage: UsageData;
  isLoading?: boolean;
  className?: string;
}

export function UsageStats({ usage, isLoading, className }: UsageStatsProps) {
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-32" />
            <Skeleton className="h-8 w-20" />
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-2">
              <Skeleton className="h-4 w-20" />
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-2 w-full" />
            </div>
            <div className="space-y-2">
              <Skeleton className="h-4 w-20" />
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-2 w-full" />
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const callsPercentage = (usage.calls_used / usage.calls_limit) * 100;
  const minutesPercentage = (usage.minutes_used / usage.minutes_limit) * 100;
  const isCallsWarning = callsPercentage >= 80;
  const isMinutesWarning = minutesPercentage >= 80;
  const isCallsCritical = callsPercentage >= 95;
  const isMinutesCritical = minutesPercentage >= 95;

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <CreditCard className="h-5 w-5" />
            Usage & Limits
          </CardTitle>
          <Button variant="ghost" size="sm" asChild>
            <Link href="/billing">Upgrade</Link>
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-2 gap-6">
          <UsageMetric
            icon={<Phone className="h-4 w-4" />}
            label="Calls"
            used={usage.calls_used}
            limit={usage.calls_limit}
            percentage={callsPercentage}
            isWarning={isCallsWarning}
            isCritical={isCallsCritical}
          />
          <UsageMetric
            icon={<Clock className="h-4 w-4" />}
            label="Minutes"
            used={usage.minutes_used}
            limit={usage.minutes_limit}
            percentage={minutesPercentage}
            isWarning={isMinutesWarning}
            isCritical={isMinutesCritical}
          />
        </div>

        {(isCallsCritical || isMinutesCritical) && (
          <div className="flex items-center gap-3 p-3 rounded-lg bg-destructive/10 border border-destructive/20">
            <AlertTriangle className="h-5 w-5 text-destructive flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-destructive">
                Usage limit almost reached
              </p>
              <p className="text-xs text-muted-foreground">
                Upgrade your plan to continue making calls
              </p>
            </div>
            <Button size="sm" variant="destructive" asChild>
              <Link href="/billing">Upgrade</Link>
            </Button>
          </div>
        )}

        {usage.billing_period_end && (
          <p className="text-xs text-muted-foreground text-center">
            Resets on {new Date(usage.billing_period_end).toLocaleDateString()}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

interface UsageMetricProps {
  icon: React.ReactNode;
  label: string;
  used: number;
  limit: number;
  percentage: number;
  isWarning: boolean;
  isCritical: boolean;
}

function UsageMetric({
  icon,
  label,
  used,
  limit,
  percentage,
  isWarning,
  isCritical,
}: UsageMetricProps) {
  const variant = isCritical ? "destructive" : isWarning ? "warning" : "default";

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          {icon}
          {label}
        </div>
        <SimpleTooltip content={`${percentage.toFixed(1)}% used`}>
          <span
            className={cn(
              "text-xs font-medium",
              isCritical
                ? "text-destructive"
                : isWarning
                ? "text-warning"
                : "text-muted-foreground"
            )}
          >
            {formatNumber(used)} / {formatNumber(limit)}
          </span>
        </SimpleTooltip>
      </div>
      <Progress value={percentage} variant={variant} size="sm" />
    </div>
  );
}

interface UsageCirclesProps {
  usage: UsageData;
  isLoading?: boolean;
  className?: string;
}

export function UsageCircles({ usage, isLoading, className }: UsageCirclesProps) {
  if (isLoading) {
    return (
      <div className={cn("flex items-center justify-center gap-8", className)}>
        <Skeleton className="h-24 w-24 rounded-full" />
        <Skeleton className="h-24 w-24 rounded-full" />
      </div>
    );
  }

  const callsPercentage = (usage.calls_used / usage.calls_limit) * 100;
  const minutesPercentage = (usage.minutes_used / usage.minutes_limit) * 100;

  return (
    <div className={cn("flex items-center justify-center gap-8", className)}>
      <div className="text-center">
        <CircularProgress
          value={callsPercentage}
          size={80}
          strokeWidth={8}
          variant={callsPercentage >= 90 ? "destructive" : callsPercentage >= 70 ? "warning" : "default"}
        />
        <p className="mt-2 text-sm font-medium">Calls</p>
        <p className="text-xs text-muted-foreground">
          {formatNumber(usage.calls_used)}/{formatNumber(usage.calls_limit)}
        </p>
      </div>
      <div className="text-center">
        <CircularProgress
          value={minutesPercentage}
          size={80}
          strokeWidth={8}
          variant={minutesPercentage >= 90 ? "destructive" : minutesPercentage >= 70 ? "warning" : "default"}
        />
        <p className="mt-2 text-sm font-medium">Minutes</p>
        <p className="text-xs text-muted-foreground">
          {formatNumber(usage.minutes_used)}/{formatNumber(usage.minutes_limit)}
        </p>
      </div>
    </div>
  );
}

interface QuickStatsProps {
  stats: {
    totalCalls: number;
    totalMinutes: number;
    successRate: number;
    activeAgents: number;
    totalAgents: number;
    trend: number;
  };
  isLoading?: boolean;
  className?: string;
}

export function QuickStats({ stats, isLoading, className }: QuickStatsProps) {
  if (isLoading) {
    return (
      <div className={cn("grid grid-cols-2 md:grid-cols-4 gap-4", className)}>
        {[1, 2, 3, 4].map((i) => (
          <Card key={i} className="p-4">
            <Skeleton className="h-4 w-20 mb-2" />
            <Skeleton className="h-8 w-16" />
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className={cn("grid grid-cols-2 md:grid-cols-4 gap-4", className)}>
      <StatCard
        label="Total Calls"
        value={formatNumber(stats.totalCalls)}
        icon={<Phone className="h-4 w-4" />}
        trend={stats.trend}
      />
      <StatCard
        label="Minutes"
        value={formatNumber(stats.totalMinutes)}
        icon={<Clock className="h-4 w-4" />}
      />
      <StatCard
        label="Success Rate"
        value={`${stats.successRate.toFixed(1)}%`}
        icon={<TrendingUp className="h-4 w-4" />}
        positive={stats.successRate >= 90}
      />
      <StatCard
        label="Active Agents"
        value={`${stats.activeAgents}/${stats.totalAgents}`}
        link="/agents"
      />
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: string;
  icon?: React.ReactNode;
  trend?: number;
  positive?: boolean;
  link?: string;
}

function StatCard({ label, value, icon, trend, positive, link }: StatCardProps) {
  const content = (
    <Card className={cn("p-4", link && "hover:border-primary/30 transition-colors cursor-pointer")}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm text-muted-foreground">{label}</span>
        {icon && <span className="text-muted-foreground">{icon}</span>}
        {link && <ArrowRight className="h-4 w-4 text-muted-foreground" />}
      </div>
      <div className="flex items-baseline gap-2">
        <span className={cn(
          "text-2xl font-bold",
          positive !== undefined && (positive ? "text-green-600" : "text-red-600")
        )}>
          {value}
        </span>
        {trend !== undefined && (
          <span
            className={cn(
              "flex items-center gap-0.5 text-xs",
              trend >= 0 ? "text-green-600" : "text-red-600"
            )}
          >
            <TrendingUp className={cn("h-3 w-3", trend < 0 && "rotate-180")} />
            {Math.abs(trend)}%
          </span>
        )}
      </div>
    </Card>
  );

  if (link) {
    return <Link href={link}>{content}</Link>;
  }

  return content;
}

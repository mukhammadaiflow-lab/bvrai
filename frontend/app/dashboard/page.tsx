"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Phone,
  Clock,
  TrendingUp,
  TrendingDown,
  Bot,
  Activity,
  PhoneCall,
  PhoneOff,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn, formatDuration, formatNumber, formatRelativeTime } from "@/lib/utils";
import { analytics, calls, agents as agentsApi } from "@/lib/api";
import Link from "next/link";

// Skeleton component for loading states
function StatCardSkeleton() {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div className="h-4 w-24 bg-muted animate-pulse rounded" />
        <div className="h-4 w-4 bg-muted animate-pulse rounded" />
      </CardHeader>
      <CardContent>
        <div className="h-8 w-20 bg-muted animate-pulse rounded mb-2" />
        <div className="h-3 w-32 bg-muted animate-pulse rounded" />
      </CardContent>
    </Card>
  );
}

function CallSkeleton() {
  return (
    <div className="flex items-center justify-between rounded-lg border p-3">
      <div className="flex items-center gap-3">
        <div className="h-10 w-10 bg-muted animate-pulse rounded-full" />
        <div>
          <div className="h-4 w-32 bg-muted animate-pulse rounded mb-1" />
          <div className="h-3 w-24 bg-muted animate-pulse rounded" />
        </div>
      </div>
      <div className="text-right">
        <div className="h-5 w-16 bg-muted animate-pulse rounded mb-1" />
        <div className="h-3 w-20 bg-muted animate-pulse rounded" />
      </div>
    </div>
  );
}

export default function DashboardPage() {
  // Fetch dashboard stats
  const { data: dashboardStats, isLoading: statsLoading, error: statsError } = useQuery({
    queryKey: ["dashboard", "stats"],
    queryFn: () => analytics.getDashboard(),
    staleTime: 60 * 1000, // 1 minute
  });

  // Fetch recent calls
  const { data: recentCallsData, isLoading: callsLoading } = useQuery({
    queryKey: ["calls", "recent"],
    queryFn: () => calls.list({ page_size: 5 }),
    staleTime: 30 * 1000, // 30 seconds
  });

  // Fetch agents for top performers
  const { data: agentsData, isLoading: agentsLoading } = useQuery({
    queryKey: ["agents", "list"],
    queryFn: () => agentsApi.list({ page_size: 4 }),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Default values for stats (used when loading or error)
  const stats = dashboardStats || {
    today: {
      total_calls: 0,
      total_minutes: 0,
      active_calls: 0,
      success_rate: 0,
    },
    week: {
      total_calls: 0,
      total_minutes: 0,
      avg_calls_per_day: 0,
      trend_percentage: 0,
    },
    agents: {
      total: 0,
      active: 0,
    },
    usage: {
      calls_used: 0,
      calls_limit: 5000,
      minutes_used: 0,
      minutes_limit: 10000,
    },
  };

  const recentCalls = recentCallsData?.items || [];
  const topAgents = agentsData?.items?.slice(0, 4) || [];

  // Error state
  if (statsError) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <AlertCircle className="h-12 w-12 text-destructive" />
        <p className="text-lg font-medium">Failed to load dashboard</p>
        <p className="text-sm text-muted-foreground">
          {statsError instanceof Error ? statsError.message : "Unknown error"}
        </p>
        <Button onClick={() => window.location.reload()}>Retry</Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {statsLoading ? (
          <>
            <StatCardSkeleton />
            <StatCardSkeleton />
            <StatCardSkeleton />
            <StatCardSkeleton />
          </>
        ) : (
          <>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Today's Calls</CardTitle>
                <Phone className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{formatNumber(stats.today.total_calls)}</div>
                <p className="text-xs text-muted-foreground">
                  {stats.today.active_calls} active right now
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Minutes</CardTitle>
                <Clock className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{formatNumber(stats.today.total_minutes)}</div>
                <p className="text-xs text-muted-foreground">
                  {stats.today.total_calls > 0
                    ? formatDuration(Math.round((stats.today.total_minutes / stats.today.total_calls) * 60))
                    : "0s"}{" "}
                  avg duration
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(stats.today.success_rate * 100).toFixed(1)}%
                </div>
                <div className="flex items-center text-xs text-green-600">
                  <TrendingUp className="mr-1 h-3 w-3" />
                  Calculated from completed calls
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
                <Bot className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {stats.agents.active}/{stats.agents.total}
                </div>
                <p className="text-xs text-muted-foreground">agents online</p>
              </CardContent>
            </Card>
          </>
        )}
      </div>

      {/* Weekly Trend */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Weekly Overview</CardTitle>
            <div
              className={cn(
                "flex items-center gap-1 text-sm",
                stats.week.trend_percentage >= 0 ? "text-green-600" : "text-red-600"
              )}
            >
              {stats.week.trend_percentage >= 0 ? (
                <TrendingUp className="h-4 w-4" />
              ) : (
                <TrendingDown className="h-4 w-4" />
              )}
              {Math.abs(stats.week.trend_percentage)}% vs last week
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Total Calls</p>
              <p className="text-2xl font-bold">{formatNumber(stats.week.total_calls)}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Total Minutes</p>
              <p className="text-2xl font-bold">{formatNumber(stats.week.total_minutes)}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Avg. Calls/Day</p>
              <p className="text-2xl font-bold">{stats.week.avg_calls_per_day}</p>
            </div>
          </div>

          {/* Usage Progress */}
          <div className="mt-6 space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted-foreground">Calls Used</span>
                <span>
                  {formatNumber(stats.usage.calls_used)} / {formatNumber(stats.usage.calls_limit)}
                </span>
              </div>
              <div className="h-2 rounded-full bg-secondary">
                <div
                  className="h-2 rounded-full bg-primary transition-all"
                  style={{
                    width: `${(stats.usage.calls_used / stats.usage.calls_limit) * 100}%`,
                  }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted-foreground">Minutes Used</span>
                <span>
                  {formatNumber(stats.usage.minutes_used)} /{" "}
                  {formatNumber(stats.usage.minutes_limit)}
                </span>
              </div>
              <div className="h-2 rounded-full bg-secondary">
                <div
                  className="h-2 rounded-full bg-primary transition-all"
                  style={{
                    width: `${(stats.usage.minutes_used / stats.usage.minutes_limit) * 100}%`,
                  }}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Two Column Layout */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Recent Calls */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Recent Calls</CardTitle>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/calls">View All</Link>
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {callsLoading ? (
                <>
                  <CallSkeleton />
                  <CallSkeleton />
                  <CallSkeleton />
                </>
              ) : recentCalls.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Phone className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No calls yet</p>
                  <p className="text-sm">Make your first call to see it here</p>
                </div>
              ) : (
                recentCalls.map((call: any) => (
                  <div
                    key={call.id}
                    className="flex items-center justify-between rounded-lg border p-3"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={cn(
                          "flex h-10 w-10 items-center justify-center rounded-full",
                          call.status === "in_progress"
                            ? "bg-green-100"
                            : call.status === "completed"
                            ? "bg-gray-100"
                            : "bg-red-100"
                        )}
                      >
                        {call.status === "in_progress" ? (
                          <PhoneCall className="h-5 w-5 text-green-600" />
                        ) : call.status === "completed" ? (
                          <Phone className="h-5 w-5 text-gray-600" />
                        ) : (
                          <PhoneOff className="h-5 w-5 text-red-600" />
                        )}
                      </div>
                      <div>
                        <p className="font-medium">{call.to_number || call.from_number}</p>
                        <p className="text-sm text-muted-foreground">
                          {call.agent_name || "Unknown Agent"}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge
                        variant={
                          call.status === "in_progress"
                            ? "success"
                            : call.status === "completed"
                            ? "secondary"
                            : "destructive"
                        }
                      >
                        {call.status === "in_progress"
                          ? "Live"
                          : call.status === "completed"
                          ? formatDuration(call.duration_seconds || 0)
                          : "Failed"}
                      </Badge>
                      <p className="text-xs text-muted-foreground mt-1">
                        {formatRelativeTime(call.created_at || call.initiated_at)}
                      </p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        {/* Top Agents */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Your Agents</CardTitle>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/agents">View All</Link>
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {agentsLoading ? (
                <>
                  <CallSkeleton />
                  <CallSkeleton />
                  <CallSkeleton />
                </>
              ) : topAgents.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Bot className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No agents yet</p>
                  <p className="text-sm">Create your first agent to get started</p>
                </div>
              ) : (
                topAgents.map((agent: any, index: number) => (
                  <div
                    key={agent.id}
                    className="flex items-center justify-between rounded-lg border p-3"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 font-bold text-primary">
                        #{index + 1}
                      </div>
                      <div>
                        <p className="font-medium">{agent.name}</p>
                        <p className="text-sm text-muted-foreground">
                          {formatNumber(agent.total_calls || 0)} calls
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge variant={agent.is_active ? "success" : "secondary"}>
                        {agent.is_active ? "Active" : "Inactive"}
                      </Badge>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            <Button asChild>
              <Link href="/agents">
                <Bot className="mr-2 h-4 w-4" />
                Create New Agent
              </Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href="/calls">
                <Phone className="mr-2 h-4 w-4" />
                View Calls
              </Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href="/analytics">
                <Activity className="mr-2 h-4 w-4" />
                View Analytics
              </Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

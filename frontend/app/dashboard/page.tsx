"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import {
  Phone,
  Clock,
  TrendingUp,
  TrendingDown,
  Bot,
  Activity,
  AlertCircle,
  RefreshCw,
  Calendar,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatCard, StatGrid } from "@/components/ui/stat-card";
import { SkeletonDashboard } from "@/components/ui/skeleton";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LiveCallBanner,
  CallVolumeChart,
  CallStatusChart,
  AgentPerformanceCard,
  RecentCallsList,
  UsageStats,
  UsageCircles,
  QuickActions,
  GettingStarted,
} from "@/components/dashboard";
import { cn, formatDuration, formatNumber } from "@/lib/utils";
import { analytics, calls, agents as agentsApi } from "@/lib/api";

// Generate chart data for the last 7 days from API data
function generateChartData(baseData: any) {
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const today = new Date().getDay();

  // Use API data if available, otherwise show empty state
  const dailyData = baseData?.daily_breakdown || [];

  return days.map((day, index) => {
    const isToday = index === (today === 0 ? 6 : today - 1);
    // Find matching day from API data
    const dayData = dailyData.find((d: any) => d.day === day);

    const callCount = dayData?.calls ?? (isToday ? (baseData?.today?.total_calls || 0) : 0);
    const minutes = dayData?.minutes ?? 0;
    const successRate = baseData?.today?.success_rate || 0.85;

    return {
      date: day,
      label: day,
      calls: callCount,
      minutes: minutes,
      success: Math.floor(callCount * successRate),
      failed: Math.floor(callCount * (1 - successRate)),
    };
  });
}

export default function DashboardPage() {
  const router = useRouter();
  const [timeRange, setTimeRange] = React.useState<"today" | "week" | "month">("today");

  // Fetch dashboard stats
  const {
    data: dashboardStats,
    isLoading: statsLoading,
    error: statsError,
    refetch: refetchStats,
  } = useQuery({
    queryKey: ["dashboard", "stats"],
    queryFn: () => analytics.getDashboard(),
    staleTime: 30 * 1000,
    refetchInterval: 60 * 1000,
  });

  // Fetch recent calls
  const { data: recentCallsData, isLoading: callsLoading } = useQuery({
    queryKey: ["calls", "recent"],
    queryFn: () => calls.list({ page_size: 6 }),
    staleTime: 15 * 1000,
    refetchInterval: 30 * 1000,
  });

  // Fetch agents for top performers
  const { data: agentsData, isLoading: agentsLoading } = useQuery({
    queryKey: ["agents", "list"],
    queryFn: () => agentsApi.list({ page_size: 6 }),
    staleTime: 2 * 60 * 1000,
  });

  // Default values for stats
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
  const topAgents = agentsData?.items?.slice(0, 6) || [];
  const chartData = generateChartData(stats);

  // Transform agents to performance format using real stats
  const agentPerformance = topAgents.map((agent: any) => ({
    id: agent.id,
    name: agent.name,
    is_active: agent.is_active ?? agent.status === "active",
    total_calls: agent.total_calls ?? agent.statistics?.total_calls ?? 0,
    successful_calls: agent.successful_calls ?? agent.statistics?.successful_calls ?? 0,
    average_duration: agent.average_duration ?? agent.statistics?.avg_duration ?? 0,
    success_rate: agent.success_rate ?? agent.statistics?.success_rate ?? 0,
    trend: agent.trend ?? agent.statistics?.trend ?? 0,
  }));

  // Determine if user is new (for getting started section)
  const isNewUser = stats.today.total_calls === 0 && stats.agents.total === 0;
  const completedSteps = [
    stats.agents.total > 0,
    topAgents.some((a: any) => a.voice_config),
    stats.today.total_calls > 0,
    stats.week.total_calls > 5,
  ].filter(Boolean).length;

  // Error state
  if (statsError) {
    return (
      <div className="space-y-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error loading dashboard</AlertTitle>
          <AlertDescription>
            {statsError instanceof Error ? statsError.message : "Failed to load dashboard data"}
          </AlertDescription>
        </Alert>
        <div className="flex justify-center">
          <Button onClick={() => refetchStats()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  // Loading state
  if (statsLoading && !dashboardStats) {
    return <SkeletonDashboard />;
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header Section */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor your voice AI performance and activity
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Tabs value={timeRange} onValueChange={(v) => setTimeRange(v as any)}>
            <TabsList>
              <TabsTrigger value="today">Today</TabsTrigger>
              <TabsTrigger value="week">This Week</TabsTrigger>
              <TabsTrigger value="month">This Month</TabsTrigger>
            </TabsList>
          </Tabs>
          <Button
            variant="outline"
            size="icon"
            onClick={() => refetchStats()}
            className="shrink-0"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Live Calls Banner */}
      {stats.today.active_calls > 0 && (
        <LiveCallBanner
          count={stats.today.active_calls}
          onViewCalls={() => router.push("/calls?status=in_progress")}
        />
      )}

      {/* Getting Started (for new users) */}
      {isNewUser && (
        <GettingStarted completedSteps={completedSteps} totalSteps={4} />
      )}

      {/* Stats Grid */}
      <StatGrid columns={4}>
        <StatCard
          title="Total Calls"
          value={formatNumber(stats.today.total_calls)}
          description={`${stats.today.active_calls} active now`}
          icon={<Phone className="h-5 w-5" />}
          trend={{
            value: stats.week.trend_percentage,
            label: "vs last week",
          }}
          href="/calls"
        />
        <StatCard
          title="Minutes Used"
          value={formatNumber(stats.today.total_minutes)}
          description={
            stats.today.total_calls > 0
              ? `${formatDuration(Math.round((stats.today.total_minutes / stats.today.total_calls) * 60))} avg`
              : "No calls yet"
          }
          icon={<Clock className="h-5 w-5" />}
        />
        <StatCard
          title="Success Rate"
          value={`${(stats.today.success_rate * 100).toFixed(1)}%`}
          description="Completed successfully"
          icon={<Activity className="h-5 w-5" />}
          trend={{
            value: 5.2,
            label: "improving",
            direction: "up",
          }}
        />
        <StatCard
          title="Active Agents"
          value={`${stats.agents.active}/${stats.agents.total}`}
          description="agents online"
          icon={<Bot className="h-5 w-5" />}
          href="/agents"
        />
      </StatGrid>

      {/* Charts Row */}
      <div className="grid gap-6 lg:grid-cols-3">
        <CallVolumeChart
          data={chartData}
          isLoading={statsLoading}
          className="lg:col-span-2"
        />
        <UsageStats
          usage={stats.usage}
          isLoading={statsLoading}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        <RecentCallsList
          calls={recentCalls}
          isLoading={callsLoading}
        />
        <AgentPerformanceCard
          agents={agentPerformance}
          isLoading={agentsLoading}
        />
      </div>

      {/* Call Status Chart */}
      <CallStatusChart
        data={chartData}
        isLoading={statsLoading}
      />

      {/* Quick Actions */}
      <QuickActions />

      {/* Weekly Summary Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Calendar className="h-5 w-5" />
                Weekly Summary
              </CardTitle>
              <CardDescription>
                Performance overview for the past 7 days
              </CardDescription>
            </div>
            <div
              className={cn(
                "flex items-center gap-1 text-sm font-medium px-2.5 py-1 rounded-full",
                stats.week.trend_percentage >= 0
                  ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                  : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
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
          <div className="grid gap-6 md:grid-cols-3">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Total Calls</p>
              <p className="text-3xl font-bold">{formatNumber(stats.week.total_calls)}</p>
              <p className="text-xs text-muted-foreground">
                ~{stats.week.avg_calls_per_day} calls per day
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Total Minutes</p>
              <p className="text-3xl font-bold">{formatNumber(stats.week.total_minutes)}</p>
              <p className="text-xs text-muted-foreground">
                {stats.week.total_calls > 0
                  ? formatDuration(Math.round((stats.week.total_minutes / stats.week.total_calls) * 60))
                  : "0s"}{" "}
                avg per call
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Usage</p>
              <UsageCircles usage={stats.usage} isLoading={statsLoading} />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

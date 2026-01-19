"use client";

import React, { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  Phone,
  Clock,
  TrendingUp,
  TrendingDown,
  Bot,
  Activity,
  DollarSign,
  Users,
  ThumbsUp,
  Target,
  Zap,
  Calendar,
  Download,
  Filter,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn, formatNumber, formatDuration } from "@/lib/utils";
import { analytics } from "@/lib/api";

// Skeleton components
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

function ChartSkeleton() {
  return (
    <div className="h-[350px] flex items-center justify-center">
      <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
    </div>
  );
}

// Default/demo data for charts when API returns empty
const defaultCallsOverTime = [
  { date: "Day 1", calls: 0, minutes: 0 },
  { date: "Day 2", calls: 0, minutes: 0 },
  { date: "Day 3", calls: 0, minutes: 0 },
  { date: "Day 4", calls: 0, minutes: 0 },
  { date: "Day 5", calls: 0, minutes: 0 },
  { date: "Day 6", calls: 0, minutes: 0 },
  { date: "Day 7", calls: 0, minutes: 0 },
];

const defaultCallsByHour = Array.from({ length: 24 }, (_, i) => ({
  hour: `${i}:00`,
  count: 0,
}));

const sentimentDistribution = [
  { name: "Positive", value: 0, color: "#22c55e" },
  { name: "Neutral", value: 0, color: "#94a3b8" },
  { name: "Negative", value: 0, color: "#ef4444" },
];

const callOutcomes = [
  { name: "Completed", value: 0, color: "#22c55e" },
  { name: "Transferred", value: 0, color: "#3b82f6" },
  { name: "No Answer", value: 0, color: "#f59e0b" },
  { name: "Failed", value: 0, color: "#ef4444" },
];

type TimeRange = "today" | "7d" | "30d" | "90d";

function getDateRange(timeRange: TimeRange): { from_date: string; to_date: string } {
  const now = new Date();
  const to_date = now.toISOString().split("T")[0];
  let from_date: string;

  switch (timeRange) {
    case "today":
      from_date = to_date;
      break;
    case "7d":
      from_date = new Date(now.setDate(now.getDate() - 7)).toISOString().split("T")[0];
      break;
    case "30d":
      from_date = new Date(now.setDate(now.getDate() - 30)).toISOString().split("T")[0];
      break;
    case "90d":
      from_date = new Date(now.setDate(now.getDate() - 90)).toISOString().split("T")[0];
      break;
  }

  return { from_date, to_date };
}

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState<TimeRange>("7d");
  const dateRange = getDateRange(timeRange);

  // Fetch analytics summary
  const { data: summaryData, isLoading: summaryLoading, error: summaryError } = useQuery({
    queryKey: ["analytics", "summary", timeRange],
    queryFn: () => analytics.getSummary(dateRange),
    staleTime: 60 * 1000, // 1 minute
  });

  // Fetch calls by day
  const { data: callsByDayData, isLoading: callsByDayLoading } = useQuery({
    queryKey: ["analytics", "calls-by-day", timeRange],
    queryFn: () => analytics.getCallsByDay(dateRange),
    staleTime: 60 * 1000,
  });

  // Fetch calls by hour
  const { data: callsByHourData, isLoading: callsByHourLoading } = useQuery({
    queryKey: ["analytics", "calls-by-hour", timeRange],
    queryFn: () => analytics.getCallsByHour(dateRange),
    staleTime: 60 * 1000,
  });

  // Fetch agent performance
  const { data: agentPerformanceData, isLoading: agentPerfLoading } = useQuery({
    queryKey: ["analytics", "agent-performance"],
    queryFn: () => analytics.getAgentPerformance(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Process data
  const summaryStats = useMemo(() => ({
    totalCalls: summaryData?.total_calls || 0,
    totalMinutes: summaryData?.total_duration_minutes || 0,
    avgDuration: summaryData?.average_duration_seconds || 0,
    successRate: summaryData?.success_rate || 0,
    avgSentiment: 0.75, // Not in summary, placeholder
    totalCost: summaryData?.total_cost_cents || 0,
    uniqueCallers: summaryData?.unique_callers || 0,
    transferRate: 0, // Not in current summary type
    activeAgents: 0, // Not in current summary type
    totalAgents: 0, // Not in current summary type
  }), [summaryData]);

  const callsOverTime = useMemo(() => {
    if (!callsByDayData || callsByDayData.length === 0) return defaultCallsOverTime;
    return callsByDayData.map((d: any) => ({
      date: new Date(d.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      calls: d.count || 0,
      minutes: d.duration || 0,
    }));
  }, [callsByDayData]);

  const callsByHour = useMemo(() => {
    if (!callsByHourData || callsByHourData.length === 0) return defaultCallsByHour;
    return callsByHourData.map((d: any) => ({
      hour: `${d.hour}:00`,
      calls: d.count || 0,
    }));
  }, [callsByHourData]);

  const agentPerformance = useMemo(() => {
    if (!agentPerformanceData || agentPerformanceData.length === 0) return [];
    return agentPerformanceData;
  }, [agentPerformanceData]);

  // Error state
  if (summaryError) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <AlertCircle className="h-12 w-12 text-destructive" />
        <p className="text-lg font-medium">Failed to load analytics</p>
        <p className="text-sm text-muted-foreground">
          {summaryError instanceof Error ? summaryError.message : "Unknown error"}
        </p>
        <Button onClick={() => window.location.reload()}>Retry</Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Analytics</h1>
          <p className="text-muted-foreground">
            Track performance, sentiment, and usage metrics
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex rounded-lg border p-1">
            {(["today", "7d", "30d", "90d"] as TimeRange[]).map((range) => (
              <Button
                key={range}
                variant={timeRange === range ? "default" : "ghost"}
                size="sm"
                onClick={() => setTimeRange(range)}
              >
                {range === "today" ? "Today" : range}
              </Button>
            ))}
          </div>
          <Button variant="outline" size="sm">
            <Filter className="mr-2 h-4 w-4" />
            Filters
          </Button>
          <Button variant="outline" size="sm">
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Summary Stats Grid */}
      {summaryLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Calls</CardTitle>
              <Phone className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(summaryStats.totalCalls)}</div>
              <div className="flex items-center text-xs text-muted-foreground">
                For selected period
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Minutes</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(summaryStats.totalMinutes)}</div>
              <p className="text-xs text-muted-foreground">
                {summaryStats.avgDuration > 0 ? formatDuration(summaryStats.avgDuration) : "0s"} avg duration
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {(summaryStats.successRate * 100).toFixed(1)}%
              </div>
              <div className="flex items-center text-xs text-muted-foreground">
                Based on completed calls
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Sentiment</CardTitle>
              <ThumbsUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={cn(
                "text-2xl font-bold",
                summaryStats.avgSentiment >= 0.6 ? "text-green-600" :
                summaryStats.avgSentiment >= 0.4 ? "text-yellow-600" : "text-red-600"
              )}>
                {(summaryStats.avgSentiment * 100).toFixed(0)}%
              </div>
              <p className="text-xs text-muted-foreground">
                Positive customer sentiment
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Calls Over Time Chart */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Calls Over Time</CardTitle>
              <CardDescription>Daily call volume and minutes</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {callsByDayLoading ? (
            <ChartSkeleton />
          ) : (
            <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={callsOverTime}>
                  <defs>
                    <linearGradient id="colorCalls" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="colorMinutes" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="date" className="text-xs" />
                  <YAxis yAxisId="left" className="text-xs" />
                  <YAxis yAxisId="right" orientation="right" className="text-xs" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Legend />
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="calls"
                    stroke="#3b82f6"
                    fillOpacity={1}
                    fill="url(#colorCalls)"
                    name="Calls"
                  />
                  <Area
                    yAxisId="right"
                    type="monotone"
                    dataKey="minutes"
                    stroke="#22c55e"
                    fillOpacity={1}
                    fill="url(#colorMinutes)"
                    name="Minutes"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Two Column Charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Calls by Hour */}
        <Card>
          <CardHeader>
            <CardTitle>Calls by Hour</CardTitle>
            <CardDescription>Peak hours distribution</CardDescription>
          </CardHeader>
          <CardContent>
            {callsByHourLoading ? (
              <div className="h-[280px] flex items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <div className="h-[280px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={callsByHour}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="hour" className="text-xs" />
                    <YAxis className="text-xs" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                      }}
                    />
                    <Bar dataKey="calls" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Sentiment Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Sentiment Distribution</CardTitle>
            <CardDescription>Customer sentiment breakdown</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center">
              <div className="h-[280px] flex-1">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={sentimentDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {sentimentDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-3">
                {sentimentDistribution.map((item) => (
                  <div key={item.name} className="flex items-center gap-3">
                    <div
                      className="h-3 w-3 rounded-full"
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-sm">{item.name}</span>
                    <span className="font-bold">{item.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Call Outcomes */}
        <Card>
          <CardHeader>
            <CardTitle>Call Outcomes</CardTitle>
            <CardDescription>How calls are resolved</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center">
              <div className="h-[280px] flex-1">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={callOutcomes}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {callOutcomes.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-3">
                {callOutcomes.map((item) => (
                  <div key={item.name} className="flex items-center gap-3">
                    <div
                      className="h-3 w-3 rounded-full"
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-sm">{item.name}</span>
                    <span className="font-bold">{formatNumber(item.value)}</span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Placeholder for future data */}
        <Card>
          <CardHeader>
            <CardTitle>Top Customer Intents</CardTitle>
            <CardDescription>Most common call purposes</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col items-center justify-center h-[280px] text-muted-foreground">
              <Activity className="h-12 w-12 mb-4 opacity-50" />
              <p>Intent analysis coming soon</p>
              <p className="text-sm">Make more calls to see intent data</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Agent Performance Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Agent Performance</CardTitle>
              <CardDescription>Performance metrics by agent</CardDescription>
            </div>
            <Button variant="outline" size="sm" asChild>
              <a href="/agents">View All</a>
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {agentPerfLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : agentPerformance.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <Bot className="h-12 w-12 mb-4 opacity-50" />
              <p>No agent data available</p>
              <p className="text-sm">Create agents and make calls to see performance</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b text-left text-sm text-muted-foreground">
                    <th className="pb-3 font-medium">Agent</th>
                    <th className="pb-3 font-medium">Total Calls</th>
                    <th className="pb-3 font-medium">Success Rate</th>
                    <th className="pb-3 font-medium">Avg Duration</th>
                    <th className="pb-3 font-medium">Grade</th>
                  </tr>
                </thead>
                <tbody>
                  {agentPerformance.map((agent: any) => {
                    const successRate = (agent.success_rate || 0) * 100;
                    const grade =
                      successRate >= 95 ? "A" :
                      successRate >= 90 ? "B" :
                      successRate >= 85 ? "C" : "D";
                    const gradeColor =
                      grade === "A" ? "bg-green-100 text-green-800" :
                      grade === "B" ? "bg-blue-100 text-blue-800" :
                      grade === "C" ? "bg-yellow-100 text-yellow-800" :
                      "bg-red-100 text-red-800";

                    return (
                      <tr key={agent.agent_id} className="border-b last:border-0">
                        <td className="py-4">
                          <div className="flex items-center gap-3">
                            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                              <Bot className="h-5 w-5 text-primary" />
                            </div>
                            <span className="font-medium">{agent.agent_name}</span>
                          </div>
                        </td>
                        <td className="py-4">{formatNumber(agent.total_calls || 0)}</td>
                        <td className="py-4">
                          <span className={cn(
                            "font-medium",
                            successRate >= 90 ? "text-green-600" : "text-yellow-600"
                          )}>
                            {successRate.toFixed(0)}%
                          </span>
                        </td>
                        <td className="py-4">{formatDuration(agent.avg_duration || 0)}</td>
                        <td className="py-4">
                          <Badge className={gradeColor}>{grade}</Badge>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Secondary Stats */}
      {summaryLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Unique Callers</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(summaryStats.uniqueCallers)}</div>
              <p className="text-xs text-muted-foreground">
                Distinct phone numbers
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Transfer Rate</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {(summaryStats.transferRate * 100).toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground">
                Calls transferred to human
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Cost</CardTitle>
              <DollarSign className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                ${(summaryStats.totalCost / 100).toFixed(2)}
              </div>
              <p className="text-xs text-muted-foreground">
                This period
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
              <Bot className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {summaryStats.activeAgents}/{summaryStats.totalAgents}
              </div>
              <p className="text-xs text-muted-foreground">
                Agents handling calls
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

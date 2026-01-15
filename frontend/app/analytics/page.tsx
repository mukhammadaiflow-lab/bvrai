"use client";

import React, { useState, useMemo } from "react";
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
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn, formatNumber, formatDuration } from "@/lib/utils";

// Mock data for demonstration
const callsOverTime = [
  { date: "Jan 8", calls: 145, minutes: 389, successRate: 0.92 },
  { date: "Jan 9", calls: 178, minutes: 456, successRate: 0.94 },
  { date: "Jan 10", calls: 156, minutes: 412, successRate: 0.91 },
  { date: "Jan 11", calls: 189, minutes: 502, successRate: 0.95 },
  { date: "Jan 12", calls: 203, minutes: 534, successRate: 0.93 },
  { date: "Jan 13", calls: 167, minutes: 445, successRate: 0.89 },
  { date: "Jan 14", calls: 195, minutes: 512, successRate: 0.96 },
];

const callsByHour = [
  { hour: "6 AM", calls: 12 },
  { hour: "7 AM", calls: 28 },
  { hour: "8 AM", calls: 45 },
  { hour: "9 AM", calls: 89 },
  { hour: "10 AM", calls: 112 },
  { hour: "11 AM", calls: 98 },
  { hour: "12 PM", calls: 67 },
  { hour: "1 PM", calls: 78 },
  { hour: "2 PM", calls: 95 },
  { hour: "3 PM", calls: 102 },
  { hour: "4 PM", calls: 87 },
  { hour: "5 PM", calls: 65 },
  { hour: "6 PM", calls: 43 },
  { hour: "7 PM", calls: 21 },
];

const sentimentDistribution = [
  { name: "Positive", value: 65, color: "#22c55e" },
  { name: "Neutral", value: 25, color: "#94a3b8" },
  { name: "Negative", value: 10, color: "#ef4444" },
];

const callOutcomes = [
  { name: "Completed", value: 847, color: "#22c55e" },
  { name: "Transferred", value: 156, color: "#3b82f6" },
  { name: "No Answer", value: 89, color: "#f59e0b" },
  { name: "Failed", value: 45, color: "#ef4444" },
];

const agentPerformance = [
  { name: "Sales Agent", calls: 523, successRate: 96, avgDuration: 245, sentiment: 0.82 },
  { name: "Support Agent", calls: 412, successRate: 92, avgDuration: 312, sentiment: 0.74 },
  { name: "Booking Agent", calls: 287, successRate: 89, avgDuration: 187, sentiment: 0.78 },
  { name: "Survey Agent", calls: 198, successRate: 95, avgDuration: 156, sentiment: 0.85 },
  { name: "Outreach Agent", calls: 156, successRate: 88, avgDuration: 203, sentiment: 0.71 },
];

const topIntents = [
  { intent: "Schedule Appointment", count: 342, percentage: 28 },
  { intent: "Product Inquiry", count: 256, percentage: 21 },
  { intent: "Support Request", count: 198, percentage: 16 },
  { intent: "Billing Question", count: 156, percentage: 13 },
  { intent: "General Inquiry", count: 134, percentage: 11 },
  { intent: "Cancellation", count: 89, percentage: 7 },
  { intent: "Complaint", count: 48, percentage: 4 },
];

const latencyMetrics = [
  { name: "STT Latency", avg: 145, p95: 250, p99: 380 },
  { name: "LLM Latency", avg: 280, p95: 450, p99: 620 },
  { name: "TTS Latency", avg: 165, p95: 290, p99: 410 },
  { name: "Total Response", avg: 590, p95: 890, p99: 1200 },
];

type TimeRange = "today" | "7d" | "30d" | "90d";

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState<TimeRange>("7d");

  const summaryStats = useMemo(() => ({
    totalCalls: 1233,
    totalMinutes: 3250,
    avgDuration: 158,
    successRate: 0.932,
    avgSentiment: 0.78,
    totalCost: 4567,
    uniqueCallers: 892,
    transferRate: 0.127,
  }), []);

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
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Calls</CardTitle>
            <Phone className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(summaryStats.totalCalls)}</div>
            <div className="flex items-center text-xs text-green-600">
              <TrendingUp className="mr-1 h-3 w-3" />
              +12.5% from last period
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
              {formatDuration(summaryStats.avgDuration)} avg duration
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
            <div className="flex items-center text-xs text-green-600">
              <TrendingUp className="mr-1 h-3 w-3" />
              +2.3% from last period
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Sentiment</CardTitle>
            <ThumbsUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {(summaryStats.avgSentiment * 100).toFixed(0)}%
            </div>
            <p className="text-xs text-muted-foreground">
              Positive customer sentiment
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Calls Over Time Chart */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Calls Over Time</CardTitle>
              <CardDescription>Daily call volume and success rate</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
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

        {/* Top Intents */}
        <Card>
          <CardHeader>
            <CardTitle>Top Customer Intents</CardTitle>
            <CardDescription>Most common call purposes</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {topIntents.map((item, index) => (
                <div key={item.intent} className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <span className="text-muted-foreground">#{index + 1}</span>
                      {item.intent}
                    </span>
                    <span className="font-medium">{item.count}</span>
                  </div>
                  <div className="h-2 rounded-full bg-secondary">
                    <div
                      className="h-2 rounded-full bg-primary transition-all"
                      style={{ width: `${item.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
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
            <Button variant="outline" size="sm">View All</Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b text-left text-sm text-muted-foreground">
                  <th className="pb-3 font-medium">Agent</th>
                  <th className="pb-3 font-medium">Total Calls</th>
                  <th className="pb-3 font-medium">Success Rate</th>
                  <th className="pb-3 font-medium">Avg Duration</th>
                  <th className="pb-3 font-medium">Sentiment</th>
                  <th className="pb-3 font-medium">Grade</th>
                </tr>
              </thead>
              <tbody>
                {agentPerformance.map((agent) => {
                  const grade =
                    agent.successRate >= 95 ? "A" :
                    agent.successRate >= 90 ? "B" :
                    agent.successRate >= 85 ? "C" : "D";
                  const gradeColor =
                    grade === "A" ? "bg-green-100 text-green-800" :
                    grade === "B" ? "bg-blue-100 text-blue-800" :
                    grade === "C" ? "bg-yellow-100 text-yellow-800" :
                    "bg-red-100 text-red-800";

                  return (
                    <tr key={agent.name} className="border-b last:border-0">
                      <td className="py-4">
                        <div className="flex items-center gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                            <Bot className="h-5 w-5 text-primary" />
                          </div>
                          <span className="font-medium">{agent.name}</span>
                        </div>
                      </td>
                      <td className="py-4">{formatNumber(agent.calls)}</td>
                      <td className="py-4">
                        <span className={cn(
                          "font-medium",
                          agent.successRate >= 90 ? "text-green-600" : "text-yellow-600"
                        )}>
                          {agent.successRate}%
                        </span>
                      </td>
                      <td className="py-4">{formatDuration(agent.avgDuration)}</td>
                      <td className="py-4">
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-24 rounded-full bg-secondary">
                            <div
                              className={cn(
                                "h-2 rounded-full",
                                agent.sentiment >= 0.75 ? "bg-green-500" :
                                agent.sentiment >= 0.5 ? "bg-yellow-500" : "bg-red-500"
                              )}
                              style={{ width: `${agent.sentiment * 100}%` }}
                            />
                          </div>
                          <span className="text-sm">{(agent.sentiment * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="py-4">
                        <Badge className={gradeColor}>{grade}</Badge>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Latency Metrics */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Response Latency</CardTitle>
              <CardDescription>System performance metrics (ms)</CardDescription>
            </div>
            <Badge variant="outline" className="text-green-600 border-green-600">
              <Activity className="mr-1 h-3 w-3" />
              All systems operational
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-4">
            {latencyMetrics.map((metric) => (
              <div key={metric.name} className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{metric.name}</span>
                  <Zap className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Average</span>
                    <span className="font-medium">{metric.avg}ms</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">P95</span>
                    <span className="font-medium">{metric.p95}ms</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">P99</span>
                    <span className={cn(
                      "font-medium",
                      metric.p99 > 500 ? "text-yellow-600" : "text-green-600"
                    )}>
                      {metric.p99}ms
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Secondary Stats */}
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
            <div className="flex items-center text-xs text-red-600">
              <TrendingDown className="mr-1 h-3 w-3" />
              -1.2% from last period
            </div>
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
              This billing period
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
            <Bot className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">5/8</div>
            <p className="text-xs text-muted-foreground">
              Agents handling calls
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

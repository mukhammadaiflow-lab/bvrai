"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
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
  Calendar,
  Download,
  TrendingUp,
  TrendingDown,
  Phone,
  Clock,
  DollarSign,
  Users,
  Bot,
  Target,
  Zap,
  Globe,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { analyticsApi } from "@/lib/api";
import { formatNumber, formatCurrency, formatDuration } from "@/lib/utils";

// Time range options
const timeRanges = [
  { value: "24h", label: "Last 24 Hours" },
  { value: "7d", label: "Last 7 Days" },
  { value: "30d", label: "Last 30 Days" },
  { value: "90d", label: "Last 90 Days" },
  { value: "1y", label: "Last Year" },
];

// Colors for charts
const COLORS = {
  primary: "#6366f1",
  secondary: "#8b5cf6",
  success: "#10b981",
  warning: "#f59e0b",
  destructive: "#ef4444",
  muted: "#6b7280",
};

const PIE_COLORS = ["#6366f1", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#06b6d4"];

// Stat Card with trend
function StatCard({
  title,
  value,
  change,
  changeLabel,
  icon: Icon,
  trend,
}: {
  title: string;
  value: string;
  change: number;
  changeLabel: string;
  icon: React.ComponentType<{ className?: string }>;
  trend: "up" | "down" | "neutral";
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        <div className="flex items-center gap-1 text-xs">
          {trend === "up" && <TrendingUp className="h-3 w-3 text-success" />}
          {trend === "down" && <TrendingDown className="h-3 w-3 text-destructive" />}
          <span
            className={
              trend === "up"
                ? "text-success"
                : trend === "down"
                ? "text-destructive"
                : "text-muted-foreground"
            }
          >
            {change > 0 ? "+" : ""}
            {change}%
          </span>
          <span className="text-muted-foreground">{changeLabel}</span>
        </div>
      </CardContent>
    </Card>
  );
}

// Call Volume Chart
function CallVolumeChart({ data }: { data: Array<{ date: string; calls: number; minutes: number }> }) {
  return (
    <Card className="col-span-full lg:col-span-2">
      <CardHeader>
        <CardTitle>Call Volume</CardTitle>
        <CardDescription>Total calls and minutes over time</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
              <defs>
                <linearGradient id="callsGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={COLORS.primary} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={COLORS.primary} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="minutesGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={COLORS.secondary} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={COLORS.secondary} stopOpacity={0} />
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
                stroke={COLORS.primary}
                fill="url(#callsGradient)"
                name="Calls"
              />
              <Area
                yAxisId="right"
                type="monotone"
                dataKey="minutes"
                stroke={COLORS.secondary}
                fill="url(#minutesGradient)"
                name="Minutes"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// Call Outcomes Chart
function CallOutcomesChart({
  data,
}: {
  data: Array<{ name: string; value: number; color: string }>;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Call Outcomes</CardTitle>
        <CardDescription>Distribution of call results</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// Agent Performance Chart
function AgentPerformanceChart({
  data,
}: {
  data: Array<{ agent: string; calls: number; success_rate: number; avg_duration: number }>;
}) {
  return (
    <Card className="col-span-full">
      <CardHeader>
        <CardTitle>Agent Performance</CardTitle>
        <CardDescription>Comparison of agent metrics</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis type="number" className="text-xs" />
              <YAxis dataKey="agent" type="category" width={120} className="text-xs" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
              />
              <Legend />
              <Bar dataKey="calls" fill={COLORS.primary} name="Total Calls" radius={[0, 4, 4, 0]} />
              <Bar
                dataKey="success_rate"
                fill={COLORS.success}
                name="Success Rate %"
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// Hourly Distribution Chart
function HourlyDistributionChart({ data }: { data: Array<{ hour: string; calls: number }> }) {
  return (
    <Card className="col-span-full lg:col-span-2">
      <CardHeader>
        <CardTitle>Call Distribution by Hour</CardTitle>
        <CardDescription>When calls are most frequent</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
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
              <Bar dataKey="calls" fill={COLORS.primary} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// Cost Analysis Chart
function CostAnalysisChart({
  data,
}: {
  data: Array<{ date: string; llm: number; voice: number; telephony: number }>;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Cost Breakdown</CardTitle>
        <CardDescription>Spending by service type</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="date" className="text-xs" />
              <YAxis className="text-xs" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
                formatter={(value: number) => formatCurrency(value)}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="llm"
                stroke={COLORS.primary}
                name="LLM"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="voice"
                stroke={COLORS.secondary}
                name="Voice"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="telephony"
                stroke={COLORS.success}
                name="Telephony"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// Geographic Distribution
function GeographicChart({
  data,
}: {
  data: Array<{ country: string; calls: number; percentage: number }>;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Geographic Distribution</CardTitle>
        <CardDescription>Calls by region</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {data.slice(0, 5).map((item) => (
            <div key={item.country} className="flex items-center gap-4">
              <Globe className="h-4 w-4 text-muted-foreground" />
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium">{item.country}</span>
                  <span className="text-sm text-muted-foreground">
                    {formatNumber(item.calls)} ({item.percentage}%)
                  </span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full"
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// Real-time Metrics
function RealTimeMetrics({
  metrics,
}: {
  metrics: {
    active_calls: number;
    calls_per_minute: number;
    avg_wait_time: number;
    queue_length: number;
  };
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-success animate-pulse" />
          <CardTitle className="text-base">Real-Time Metrics</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-2xl font-bold">{metrics.active_calls}</p>
            <p className="text-xs text-muted-foreground">Active Calls</p>
          </div>
          <div>
            <p className="text-2xl font-bold">{metrics.calls_per_minute}</p>
            <p className="text-xs text-muted-foreground">Calls/Min</p>
          </div>
          <div>
            <p className="text-2xl font-bold">{formatDuration(metrics.avg_wait_time)}</p>
            <p className="text-xs text-muted-foreground">Avg Wait</p>
          </div>
          <div>
            <p className="text-2xl font-bold">{metrics.queue_length}</p>
            <p className="text-xs text-muted-foreground">In Queue</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState("7d");

  // Fetch analytics data
  const { data: overview } = useQuery({
    queryKey: ["analytics-overview", timeRange],
    queryFn: () => analyticsApi.getOverview({ time_range: timeRange }),
  });

  const { data: callVolume } = useQuery({
    queryKey: ["analytics-call-volume", timeRange],
    queryFn: () => analyticsApi.getCallVolume({ time_range: timeRange }),
  });

  const { data: agentPerformance } = useQuery({
    queryKey: ["analytics-agent-performance", timeRange],
    queryFn: () => analyticsApi.getAgentPerformance({ time_range: timeRange }),
  });

  // Mock data for demonstration
  const mockCallVolume = [
    { date: "Mon", calls: 120, minutes: 450 },
    { date: "Tue", calls: 145, minutes: 520 },
    { date: "Wed", calls: 132, minutes: 480 },
    { date: "Thu", calls: 168, minutes: 610 },
    { date: "Fri", calls: 155, minutes: 580 },
    { date: "Sat", calls: 89, minutes: 320 },
    { date: "Sun", calls: 76, minutes: 280 },
  ];

  const mockOutcomes = [
    { name: "Successful", value: 68, color: COLORS.success },
    { name: "Transferred", value: 15, color: COLORS.primary },
    { name: "Voicemail", value: 10, color: COLORS.warning },
    { name: "Failed", value: 7, color: COLORS.destructive },
  ];

  const mockAgentPerformance = [
    { agent: "Sales Assistant", calls: 245, success_rate: 92, avg_duration: 180 },
    { agent: "Support Bot", calls: 189, success_rate: 88, avg_duration: 240 },
    { agent: "Appointment Scheduler", calls: 156, success_rate: 95, avg_duration: 120 },
    { agent: "Lead Qualifier", calls: 134, success_rate: 78, avg_duration: 150 },
  ];

  const mockHourlyData = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i}:00`,
    calls: Math.floor(Math.random() * 50) + 10,
  }));

  const mockCostData = [
    { date: "Week 1", llm: 120, voice: 80, telephony: 45 },
    { date: "Week 2", llm: 135, voice: 95, telephony: 52 },
    { date: "Week 3", llm: 115, voice: 88, telephony: 48 },
    { date: "Week 4", llm: 148, voice: 102, telephony: 58 },
  ];

  const mockGeoData = [
    { country: "United States", calls: 4520, percentage: 45 },
    { country: "United Kingdom", calls: 1820, percentage: 18 },
    { country: "Canada", calls: 1240, percentage: 12 },
    { country: "Australia", calls: 980, percentage: 10 },
    { country: "Germany", calls: 640, percentage: 6 },
  ];

  const mockRealTime = {
    active_calls: 23,
    calls_per_minute: 4.2,
    avg_wait_time: 12,
    queue_length: 8,
  };

  const handleExport = () => {
    // Export analytics data
    console.log("Exporting analytics...");
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold">Analytics</h1>
          <p className="text-muted-foreground">
            Comprehensive insights into your voice AI performance
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[180px]">
              <Calendar className="mr-2 h-4 w-4" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {timeRanges.map((range) => (
                <SelectItem key={range.value} value={range.value}>
                  {range.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={handleExport}>
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Calls"
          value={formatNumber(overview?.total_calls || 8945)}
          change={12.5}
          changeLabel="vs last period"
          icon={Phone}
          trend="up"
        />
        <StatCard
          title="Total Minutes"
          value={formatNumber(overview?.total_minutes || 32450)}
          change={8.2}
          changeLabel="vs last period"
          icon={Clock}
          trend="up"
        />
        <StatCard
          title="Revenue Generated"
          value={formatCurrency(overview?.revenue || 12450)}
          change={15.3}
          changeLabel="vs last period"
          icon={DollarSign}
          trend="up"
        />
        <StatCard
          title="Success Rate"
          value={`${overview?.success_rate || 94.2}%`}
          change={2.1}
          changeLabel="vs last period"
          icon={Target}
          trend="up"
        />
      </div>

      {/* Secondary Stats */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Active Agents"
          value={String(overview?.active_agents || 12)}
          change={3}
          changeLabel="new this week"
          icon={Bot}
          trend="up"
        />
        <StatCard
          title="Avg Call Duration"
          value={formatDuration(overview?.avg_duration || 185)}
          change={-5.2}
          changeLabel="vs last period"
          icon={Clock}
          trend="down"
        />
        <StatCard
          title="Unique Callers"
          value={formatNumber(overview?.unique_callers || 3240)}
          change={18.7}
          changeLabel="vs last period"
          icon={Users}
          trend="up"
        />
        <StatCard
          title="Automations Triggered"
          value={formatNumber(overview?.automations || 1856)}
          change={25.4}
          changeLabel="vs last period"
          icon={Zap}
          trend="up"
        />
      </div>

      {/* Charts Row 1 */}
      <div className="grid gap-6 lg:grid-cols-3">
        <CallVolumeChart data={callVolume?.data || mockCallVolume} />
        <CallOutcomesChart data={mockOutcomes} />
      </div>

      {/* Charts Row 2 */}
      <div className="grid gap-6 lg:grid-cols-3">
        <HourlyDistributionChart data={mockHourlyData} />
        <RealTimeMetrics metrics={mockRealTime} />
      </div>

      {/* Charts Row 3 */}
      <div className="grid gap-6 lg:grid-cols-2">
        <CostAnalysisChart data={mockCostData} />
        <GeographicChart data={mockGeoData} />
      </div>

      {/* Agent Performance */}
      <AgentPerformanceChart data={agentPerformance?.data || mockAgentPerformance} />
    </div>
  );
}

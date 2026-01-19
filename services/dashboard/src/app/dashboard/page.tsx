"use client";

import { useQuery } from "@tanstack/react-query";
import {
  PhoneCall,
  Clock,
  Bot,
  TrendingUp,
  ArrowUpRight,
  ArrowDownRight,
  Phone,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { analyticsApi, agentsApi, callsApi } from "@/lib/api";
import { formatNumber, formatDuration, formatRelativeTime } from "@/lib/utils";

// Stats Card Component
function StatsCard({
  title,
  value,
  change,
  changeType,
  icon: Icon,
  description,
}: {
  title: string;
  value: string;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon: React.ComponentType<{ className?: string }>;
  description?: string;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {change && (
          <p className="text-xs text-muted-foreground">
            <span
              className={
                changeType === "positive"
                  ? "text-success"
                  : changeType === "negative"
                  ? "text-destructive"
                  : ""
              }
            >
              {changeType === "positive" && <ArrowUpRight className="inline h-3 w-3" />}
              {changeType === "negative" && <ArrowDownRight className="inline h-3 w-3" />}
              {change}
            </span>
            {description && ` ${description}`}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

// Recent Calls Table
function RecentCalls({ calls }: { calls: unknown[] }) {
  return (
    <Card className="col-span-full lg:col-span-2">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Recent Calls</CardTitle>
            <CardDescription>Latest voice interactions</CardDescription>
          </div>
          <Button variant="outline" size="sm">
            View All
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {calls.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No calls yet. Start by creating an agent and making your first call.
            </div>
          ) : (
            calls.slice(0, 5).map((call: unknown, i: number) => {
              const c = call as {
                id: string;
                phone_number: string;
                direction: string;
                duration_seconds: number;
                status: string;
                started_at: string;
                agent_name?: string;
              };
              return (
                <div key={c.id || i} className="flex items-center gap-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                    <Phone className="h-5 w-5 text-primary" />
                  </div>
                  <div className="flex-1 space-y-1">
                    <p className="text-sm font-medium">{c.phone_number}</p>
                    <p className="text-xs text-muted-foreground">
                      {c.direction} • {c.agent_name || "Unknown Agent"}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm">{formatDuration(c.duration_seconds)}</p>
                    <p className="text-xs text-muted-foreground">
                      {formatRelativeTime(c.started_at)}
                    </p>
                  </div>
                  <div
                    className={`h-2 w-2 rounded-full ${
                      c.status === "completed"
                        ? "bg-success"
                        : c.status === "active"
                        ? "bg-primary animate-pulse"
                        : "bg-destructive"
                    }`}
                  />
                </div>
              );
            })
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// Active Agents
function ActiveAgents({ agents }: { agents: unknown[] }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Active Agents</CardTitle>
            <CardDescription>Currently deployed agents</CardDescription>
          </div>
          <Button variant="outline" size="sm">
            Manage
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {agents.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No agents yet. Create your first AI agent to get started.
            </div>
          ) : (
            agents.slice(0, 5).map((agent: unknown, i: number) => {
              const a = agent as {
                id: string;
                name: string;
                status: string;
                calls_today?: number;
              };
              return (
                <div key={a.id || i} className="flex items-center gap-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-secondary">
                    <Bot className="h-5 w-5" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium">{a.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {a.calls_today || 0} calls today
                    </p>
                  </div>
                  <div
                    className={`flex items-center gap-1.5 rounded-full px-2 py-1 text-xs font-medium ${
                      a.status === "active"
                        ? "bg-success/10 text-success"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    <div
                      className={`h-1.5 w-1.5 rounded-full ${
                        a.status === "active" ? "bg-success" : "bg-muted-foreground"
                      }`}
                    />
                    {a.status}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default function DashboardPage() {
  // Fetch dashboard data
  const { data: stats } = useQuery({
    queryKey: ["dashboard-stats"],
    queryFn: () => analyticsApi.getDashboard(),
  });

  const { data: agentsData } = useQuery({
    queryKey: ["agents-list"],
    queryFn: () => agentsApi.list({ page_size: 5 }),
  });

  const { data: callsData } = useQuery({
    queryKey: ["recent-calls"],
    queryFn: () => callsApi.list({ page_size: 5 }),
  });

  const agents = agentsData?.agents || [];
  const calls = callsData?.calls || [];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground">
          Welcome back! Here&apos;s an overview of your voice AI platform.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="Total Calls"
          value={formatNumber(stats?.total_calls || 0)}
          change="+12%"
          changeType="positive"
          description="from last month"
          icon={PhoneCall}
        />
        <StatsCard
          title="Minutes Used"
          value={formatNumber(Math.round(stats?.total_minutes || 0))}
          change="+8%"
          changeType="positive"
          description="from last month"
          icon={Clock}
        />
        <StatsCard
          title="Active Agents"
          value={String(stats?.active_agents || agents.filter((a: unknown) => (a as { status: string }).status === "active").length)}
          change="2 new"
          changeType="neutral"
          description="this week"
          icon={Bot}
        />
        <StatsCard
          title="Success Rate"
          value={`${stats?.success_rate || 98}%`}
          change="+2.5%"
          changeType="positive"
          description="from last week"
          icon={TrendingUp}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        <RecentCalls calls={calls} />
        <ActiveAgents agents={agents} />
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>Common tasks to get you started</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Button variant="outline" className="h-auto flex-col gap-2 p-4">
              <Bot className="h-6 w-6" />
              <span>Create Agent</span>
            </Button>
            <Button variant="outline" className="h-auto flex-col gap-2 p-4">
              <Phone className="h-6 w-6" />
              <span>Get Phone Number</span>
            </Button>
            <Button variant="outline" className="h-auto flex-col gap-2 p-4">
              <PhoneCall className="h-6 w-6" />
              <span>Make Test Call</span>
            </Button>
            <Button variant="outline" className="h-auto flex-col gap-2 p-4">
              <TrendingUp className="h-6 w-6" />
              <span>View Analytics</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

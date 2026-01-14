"use client";

import React from "react";
import {
  Phone,
  Clock,
  TrendingUp,
  TrendingDown,
  Bot,
  Activity,
  PhoneCall,
  PhoneOff,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn, formatDuration, formatNumber, formatRelativeTime } from "@/lib/utils";

// Mock data for demonstration
const stats = {
  today: {
    totalCalls: 156,
    totalMinutes: 432,
    activeCalls: 3,
    successRate: 0.94,
  },
  week: {
    totalCalls: 1247,
    totalMinutes: 3842,
    avgCallsPerDay: 178,
    trendPercentage: 12.5,
  },
  agents: {
    total: 8,
    active: 5,
  },
  usage: {
    callsUsed: 1247,
    callsLimit: 5000,
    minutesUsed: 3842,
    minutesLimit: 10000,
  },
};

const recentCalls = [
  {
    id: "1",
    agentName: "Sales Agent",
    phone: "+1 (555) 123-4567",
    duration: 245,
    status: "completed",
    time: "2024-01-14T10:30:00Z",
  },
  {
    id: "2",
    agentName: "Support Agent",
    phone: "+1 (555) 987-6543",
    duration: 0,
    status: "in_progress",
    time: "2024-01-14T10:35:00Z",
  },
  {
    id: "3",
    agentName: "Sales Agent",
    phone: "+1 (555) 456-7890",
    duration: 180,
    status: "completed",
    time: "2024-01-14T10:15:00Z",
  },
  {
    id: "4",
    agentName: "Booking Agent",
    phone: "+1 (555) 321-0987",
    duration: 0,
    status: "failed",
    time: "2024-01-14T10:10:00Z",
  },
];

const topAgents = [
  { name: "Sales Agent", calls: 523, successRate: 0.96 },
  { name: "Support Agent", calls: 412, successRate: 0.92 },
  { name: "Booking Agent", calls: 187, successRate: 0.89 },
  { name: "Survey Agent", calls: 125, successRate: 0.95 },
];

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Today's Calls</CardTitle>
            <Phone className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(stats.today.totalCalls)}</div>
            <p className="text-xs text-muted-foreground">
              {stats.today.activeCalls} active right now
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Minutes</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(stats.today.totalMinutes)}</div>
            <p className="text-xs text-muted-foreground">
              {formatDuration(Math.round(stats.today.totalMinutes / stats.today.totalCalls * 60))} avg duration
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(stats.today.successRate * 100).toFixed(1)}%</div>
            <div className="flex items-center text-xs text-green-600">
              <TrendingUp className="mr-1 h-3 w-3" />
              +2.1% from yesterday
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
      </div>

      {/* Weekly Trend */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Weekly Overview</CardTitle>
            <div className={cn(
              "flex items-center gap-1 text-sm",
              stats.week.trendPercentage >= 0 ? "text-green-600" : "text-red-600"
            )}>
              {stats.week.trendPercentage >= 0 ? (
                <TrendingUp className="h-4 w-4" />
              ) : (
                <TrendingDown className="h-4 w-4" />
              )}
              {Math.abs(stats.week.trendPercentage)}% vs last week
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Total Calls</p>
              <p className="text-2xl font-bold">{formatNumber(stats.week.totalCalls)}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Total Minutes</p>
              <p className="text-2xl font-bold">{formatNumber(stats.week.totalMinutes)}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Avg. Calls/Day</p>
              <p className="text-2xl font-bold">{stats.week.avgCallsPerDay}</p>
            </div>
          </div>

          {/* Usage Progress */}
          <div className="mt-6 space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted-foreground">Calls Used</span>
                <span>{formatNumber(stats.usage.callsUsed)} / {formatNumber(stats.usage.callsLimit)}</span>
              </div>
              <div className="h-2 rounded-full bg-secondary">
                <div
                  className="h-2 rounded-full bg-primary transition-all"
                  style={{ width: `${(stats.usage.callsUsed / stats.usage.callsLimit) * 100}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted-foreground">Minutes Used</span>
                <span>{formatNumber(stats.usage.minutesUsed)} / {formatNumber(stats.usage.minutesLimit)}</span>
              </div>
              <div className="h-2 rounded-full bg-secondary">
                <div
                  className="h-2 rounded-full bg-primary transition-all"
                  style={{ width: `${(stats.usage.minutesUsed / stats.usage.minutesLimit) * 100}%` }}
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
              <Button variant="ghost" size="sm">View All</Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentCalls.map((call) => (
                <div
                  key={call.id}
                  className="flex items-center justify-between rounded-lg border p-3"
                >
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      "flex h-10 w-10 items-center justify-center rounded-full",
                      call.status === "in_progress" ? "bg-green-100" :
                      call.status === "completed" ? "bg-gray-100" :
                      "bg-red-100"
                    )}>
                      {call.status === "in_progress" ? (
                        <PhoneCall className="h-5 w-5 text-green-600" />
                      ) : call.status === "completed" ? (
                        <Phone className="h-5 w-5 text-gray-600" />
                      ) : (
                        <PhoneOff className="h-5 w-5 text-red-600" />
                      )}
                    </div>
                    <div>
                      <p className="font-medium">{call.phone}</p>
                      <p className="text-sm text-muted-foreground">{call.agentName}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <Badge
                      variant={
                        call.status === "in_progress" ? "success" :
                        call.status === "completed" ? "secondary" :
                        "destructive"
                      }
                    >
                      {call.status === "in_progress" ? "Live" :
                       call.status === "completed" ? formatDuration(call.duration) :
                       "Failed"}
                    </Badge>
                    <p className="text-xs text-muted-foreground mt-1">
                      {formatRelativeTime(call.time)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Top Agents */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Top Performing Agents</CardTitle>
              <Button variant="ghost" size="sm">View All</Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {topAgents.map((agent, index) => (
                <div
                  key={agent.name}
                  className="flex items-center justify-between rounded-lg border p-3"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 font-bold text-primary">
                      #{index + 1}
                    </div>
                    <div>
                      <p className="font-medium">{agent.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {formatNumber(agent.calls)} calls this week
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-bold text-green-600">
                      {(agent.successRate * 100).toFixed(0)}%
                    </p>
                    <p className="text-xs text-muted-foreground">success rate</p>
                  </div>
                </div>
              ))}
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
            <Button>
              <Bot className="mr-2 h-4 w-4" />
              Create New Agent
            </Button>
            <Button variant="outline">
              <Phone className="mr-2 h-4 w-4" />
              Make Test Call
            </Button>
            <Button variant="outline">
              <Activity className="mr-2 h-4 w-4" />
              View Analytics
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

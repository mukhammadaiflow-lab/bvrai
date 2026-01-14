"use client";

import React, { useState } from "react";
import {
  Phone,
  PhoneIncoming,
  PhoneOutgoing,
  PhoneOff,
  Search,
  Filter,
  Download,
  Play,
  Eye,
  MoreHorizontal,
  Calendar,
  Clock,
  Bot,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  cn,
  formatDate,
  formatDuration,
  formatPhoneNumber,
  formatRelativeTime,
  getStatusColor,
} from "@/lib/utils";

// Mock calls data
const mockCalls = [
  {
    id: "call_1",
    agentName: "Sales Agent",
    agentId: "agent_1",
    direction: "inbound",
    fromNumber: "+15551234567",
    toNumber: "+15559876543",
    status: "completed",
    startedAt: "2024-01-14T10:30:00Z",
    endedAt: "2024-01-14T10:34:05Z",
    duration: 245,
    recordingUrl: "https://example.com/recording1.mp3",
    costCents: 12,
    sentiment: "positive",
    outcome: "qualified_lead",
  },
  {
    id: "call_2",
    agentName: "Support Agent",
    agentId: "agent_2",
    direction: "inbound",
    fromNumber: "+15559876543",
    toNumber: "+15551234567",
    status: "in_progress",
    startedAt: "2024-01-14T10:35:00Z",
    endedAt: null,
    duration: null,
    recordingUrl: null,
    costCents: null,
    sentiment: null,
    outcome: null,
  },
  {
    id: "call_3",
    agentName: "Sales Agent",
    agentId: "agent_1",
    direction: "outbound",
    fromNumber: "+15551234567",
    toNumber: "+15554567890",
    status: "completed",
    startedAt: "2024-01-14T10:15:00Z",
    endedAt: "2024-01-14T10:18:00Z",
    duration: 180,
    recordingUrl: "https://example.com/recording2.mp3",
    costCents: 9,
    sentiment: "neutral",
    outcome: "callback_scheduled",
  },
  {
    id: "call_4",
    agentName: "Booking Agent",
    agentId: "agent_3",
    direction: "inbound",
    fromNumber: "+15553210987",
    toNumber: "+15551234567",
    status: "failed",
    startedAt: "2024-01-14T10:10:00Z",
    endedAt: "2024-01-14T10:10:05Z",
    duration: 5,
    recordingUrl: null,
    costCents: 0,
    sentiment: null,
    outcome: "no_answer",
  },
  {
    id: "call_5",
    agentName: "Survey Agent",
    agentId: "agent_4",
    direction: "outbound",
    fromNumber: "+15551234567",
    toNumber: "+15557890123",
    status: "completed",
    startedAt: "2024-01-14T09:45:00Z",
    endedAt: "2024-01-14T09:48:30Z",
    duration: 210,
    recordingUrl: "https://example.com/recording3.mp3",
    costCents: 11,
    sentiment: "positive",
    outcome: "survey_completed",
  },
];

const statusOptions = ["all", "in_progress", "completed", "failed"];
const directionOptions = ["all", "inbound", "outbound"];

export default function CallsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [directionFilter, setDirectionFilter] = useState("all");

  const filteredCalls = mockCalls.filter((call) => {
    const matchesSearch =
      call.fromNumber.includes(searchQuery) ||
      call.toNumber.includes(searchQuery) ||
      call.agentName.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === "all" || call.status === statusFilter;
    const matchesDirection = directionFilter === "all" || call.direction === directionFilter;
    return matchesSearch && matchesStatus && matchesDirection;
  });

  // Stats
  const stats = {
    total: mockCalls.length,
    active: mockCalls.filter((c) => c.status === "in_progress").length,
    completed: mockCalls.filter((c) => c.status === "completed").length,
    failed: mockCalls.filter((c) => c.status === "failed").length,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-bold">Call Logs</h1>
          <p className="text-muted-foreground">
            View and analyze all voice agent calls
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline">
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
          <Button>
            <Phone className="mr-2 h-4 w-4" />
            New Call
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Calls</p>
                <p className="text-2xl font-bold">{stats.total}</p>
              </div>
              <Phone className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Active Now</p>
                <p className="text-2xl font-bold text-green-600">{stats.active}</p>
              </div>
              <div className="relative">
                <Phone className="h-8 w-8 text-green-600" />
                {stats.active > 0 && (
                  <span className="absolute -right-1 -top-1 flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                  </span>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Completed</p>
                <p className="text-2xl font-bold">{stats.completed}</p>
              </div>
              <Phone className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Failed</p>
                <p className="text-2xl font-bold text-red-600">{stats.failed}</p>
              </div>
              <PhoneOff className="h-8 w-8 text-red-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-center">
            <div className="flex-1">
              <Input
                placeholder="Search by phone number or agent..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                leftIcon={<Search className="h-4 w-4" />}
                className="max-w-md"
              />
            </div>
            <div className="flex gap-2">
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="all">All Status</option>
                <option value="in_progress">In Progress</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>
              <select
                value={directionFilter}
                onChange={(e) => setDirectionFilter(e.target.value)}
                className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="all">All Direction</option>
                <option value="inbound">Inbound</option>
                <option value="outbound">Outbound</option>
              </select>
              <Button variant="outline" size="icon">
                <Calendar className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Calls Table */}
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    Call
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    Agent
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    Duration
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    Status
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    Sentiment
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    Time
                  </th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-muted-foreground">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredCalls.map((call) => (
                  <tr key={call.id} className="border-b hover:bg-muted/50">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        <div className={cn(
                          "flex h-10 w-10 items-center justify-center rounded-full",
                          call.direction === "inbound" ? "bg-blue-100" : "bg-green-100"
                        )}>
                          {call.direction === "inbound" ? (
                            <PhoneIncoming className="h-5 w-5 text-blue-600" />
                          ) : (
                            <PhoneOutgoing className="h-5 w-5 text-green-600" />
                          )}
                        </div>
                        <div>
                          <p className="font-medium">
                            {formatPhoneNumber(
                              call.direction === "inbound" ? call.fromNumber : call.toNumber
                            )}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            {call.direction === "inbound" ? "Inbound" : "Outbound"}
                          </p>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <Bot className="h-4 w-4 text-muted-foreground" />
                        <span>{call.agentName}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4 text-muted-foreground" />
                        <span>
                          {call.status === "in_progress" ? (
                            <span className="text-green-600">Live</span>
                          ) : (
                            formatDuration(call.duration)
                          )}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <Badge className={getStatusColor(call.status)}>
                        {call.status.replace("_", " ")}
                      </Badge>
                    </td>
                    <td className="px-4 py-3">
                      {call.sentiment && (
                        <Badge
                          variant={
                            call.sentiment === "positive" ? "success" :
                            call.sentiment === "negative" ? "destructive" :
                            "secondary"
                          }
                        >
                          {call.sentiment}
                        </Badge>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <div>
                        <p className="text-sm">{formatDate(call.startedAt, "PP")}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatDate(call.startedAt, "p")}
                        </p>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-1">
                        {call.recordingUrl && (
                          <Button variant="ghost" size="icon-sm" title="Play Recording">
                            <Play className="h-4 w-4" />
                          </Button>
                        )}
                        <Button variant="ghost" size="icon-sm" title="View Details">
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button variant="ghost" size="icon-sm">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between px-4 py-3 border-t">
            <p className="text-sm text-muted-foreground">
              Showing {filteredCalls.length} of {mockCalls.length} calls
            </p>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" disabled>
                Previous
              </Button>
              <Button variant="outline" size="sm" disabled>
                Next
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

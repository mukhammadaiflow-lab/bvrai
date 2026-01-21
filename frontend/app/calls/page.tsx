"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
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
  Loader2,
  AlertCircle,
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
import { calls } from "@/lib/api";

// Skeleton for table rows
function CallRowSkeleton() {
  return (
    <tr className="border-b">
      <td className="px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 bg-muted animate-pulse rounded-full" />
          <div>
            <div className="h-4 w-28 bg-muted animate-pulse rounded mb-1" />
            <div className="h-3 w-16 bg-muted animate-pulse rounded" />
          </div>
        </div>
      </td>
      <td className="px-4 py-3">
        <div className="h-4 w-24 bg-muted animate-pulse rounded" />
      </td>
      <td className="px-4 py-3">
        <div className="h-4 w-16 bg-muted animate-pulse rounded" />
      </td>
      <td className="px-4 py-3">
        <div className="h-5 w-20 bg-muted animate-pulse rounded" />
      </td>
      <td className="px-4 py-3">
        <div className="h-5 w-16 bg-muted animate-pulse rounded" />
      </td>
      <td className="px-4 py-3">
        <div className="h-4 w-20 bg-muted animate-pulse rounded mb-1" />
        <div className="h-3 w-16 bg-muted animate-pulse rounded" />
      </td>
      <td className="px-4 py-3">
        <div className="h-8 w-24 bg-muted animate-pulse rounded" />
      </td>
    </tr>
  );
}

// Skeleton for stats card
function StatCardSkeleton() {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="h-4 w-20 bg-muted animate-pulse rounded mb-2" />
            <div className="h-8 w-12 bg-muted animate-pulse rounded" />
          </div>
          <div className="h-8 w-8 bg-muted animate-pulse rounded" />
        </div>
      </CardContent>
    </Card>
  );
}

export default function CallsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [directionFilter, setDirectionFilter] = useState("all");
  const [page, setPage] = useState(1);
  const pageSize = 20;

  // Fetch calls from API
  const { data: callsData, isLoading, error } = useQuery({
    queryKey: ["calls", "list", page, statusFilter, directionFilter],
    queryFn: () => calls.list({
      page,
      page_size: pageSize,
      status: statusFilter !== "all" ? statusFilter : undefined,
      direction: directionFilter !== "all" ? directionFilter : undefined,
    }),
    staleTime: 15 * 1000, // 15 seconds
  });

  const allCalls = callsData?.items || [];
  const totalCalls = callsData?.total || 0;
  const totalPages = Math.ceil(totalCalls / pageSize);

  // Client-side search filtering
  const filteredCalls = allCalls.filter((call: any) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      (call.from_number && call.from_number.includes(searchQuery)) ||
      (call.to_number && call.to_number.includes(searchQuery)) ||
      (call.agent_name && call.agent_name.toLowerCase().includes(query))
    );
  });

  // Stats - computed from visible data
  const stats = {
    total: totalCalls,
    active: allCalls.filter((c: any) => c.status === "in_progress").length,
    completed: allCalls.filter((c: any) => c.status === "completed").length,
    failed: allCalls.filter((c: any) => c.status === "failed").length,
  };

  // Error state
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <AlertCircle className="h-12 w-12 text-destructive" />
        <p className="text-lg font-medium">Failed to load calls</p>
        <p className="text-sm text-muted-foreground">
          {error instanceof Error ? error.message : "Unknown error"}
        </p>
        <Button onClick={() => window.location.reload()}>Retry</Button>
      </div>
    );
  }

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
        <div className="flex gap-2 flex-wrap">
          <Button variant="outline" className="whitespace-nowrap">
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
          <Button asChild className="whitespace-nowrap">
            <Link href="/calls/new">
              <Phone className="mr-2 h-4 w-4" />
              New Call
            </Link>
          </Button>
        </div>
      </div>

      {/* Stats */}
      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-4">
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
        </div>
      ) : (
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
      )}

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
                onChange={(e) => {
                  setStatusFilter(e.target.value);
                  setPage(1);
                }}
                className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="all">All Status</option>
                <option value="in_progress">In Progress</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>
              <select
                value={directionFilter}
                onChange={(e) => {
                  setDirectionFilter(e.target.value);
                  setPage(1);
                }}
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
                {isLoading ? (
                  <>
                    <CallRowSkeleton />
                    <CallRowSkeleton />
                    <CallRowSkeleton />
                    <CallRowSkeleton />
                    <CallRowSkeleton />
                  </>
                ) : filteredCalls.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-4 py-12 text-center">
                      <Phone className="h-8 w-8 mx-auto mb-2 text-muted-foreground opacity-50" />
                      <p className="text-muted-foreground">No calls found</p>
                      {searchQuery && (
                        <p className="text-sm text-muted-foreground mt-1">
                          Try adjusting your search
                        </p>
                      )}
                    </td>
                  </tr>
                ) : (
                  filteredCalls.map((call: any) => (
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
                                call.direction === "inbound" ? call.from_number : call.to_number
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
                          <span>{call.agent_name || "Unknown"}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <Clock className="h-4 w-4 text-muted-foreground" />
                          <span>
                            {call.status === "in_progress" ? (
                              <span className="text-green-600">Live</span>
                            ) : (
                              formatDuration(call.duration_seconds || 0)
                            )}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <Badge className={getStatusColor(call.status)}>
                          {call.status?.replace("_", " ") || "unknown"}
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
                          <p className="text-sm">{formatDate(call.created_at || call.initiated_at, "PP")}</p>
                          <p className="text-xs text-muted-foreground">
                            {formatDate(call.created_at || call.initiated_at, "p")}
                          </p>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="flex items-center justify-end gap-1">
                          {call.recording_url && (
                            <Button variant="ghost" size="icon-sm" title="Play Recording">
                              <Play className="h-4 w-4" />
                            </Button>
                          )}
                          <Button variant="ghost" size="icon-sm" title="View Details" asChild>
                            <Link href={`/calls/${call.id}`}>
                              <Eye className="h-4 w-4" />
                            </Link>
                          </Button>
                          <Button variant="ghost" size="icon-sm">
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between px-4 py-3 border-t">
            <p className="text-sm text-muted-foreground">
              {isLoading ? (
                "Loading..."
              ) : (
                `Showing ${(page - 1) * pageSize + 1}-${Math.min(page * pageSize, totalCalls)} of ${totalCalls} calls`
              )}
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page <= 1 || isLoading}
                onClick={() => setPage(p => p - 1)}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={page >= totalPages || isLoading}
                onClick={() => setPage(p => p + 1)}
              >
                Next
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

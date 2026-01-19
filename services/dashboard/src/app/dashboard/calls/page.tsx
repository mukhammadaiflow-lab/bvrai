"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  PhoneCall,
  PhoneIncoming,
  PhoneOutgoing,
  Search,
  Filter,
  Download,
  Play,
  FileText,
  Clock,
  User,
  Bot,
  Calendar,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { callsApi, Call } from "@/lib/api";
import { formatDuration, formatRelativeTime, formatPhoneNumber, cn } from "@/lib/utils";

// Call Details Dialog
function CallDetailsDialog({
  call,
  open,
  onOpenChange,
}: {
  call: Call | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { data: transcript } = useQuery({
    queryKey: ["call-transcript", call?.id],
    queryFn: () => (call ? callsApi.getTranscript(call.id) : null),
    enabled: !!call && open,
  });

  if (!call) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {call.direction === "inbound" ? (
              <PhoneIncoming className="h-5 w-5 text-primary" />
            ) : (
              <PhoneOutgoing className="h-5 w-5 text-primary" />
            )}
            Call Details
          </DialogTitle>
          <DialogDescription>
            {formatPhoneNumber(call.phone_number)} • {formatRelativeTime(call.started_at)}
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-6 py-4 overflow-y-auto">
          {/* Call Info */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">Duration</p>
              <p className="font-medium">{formatDuration(call.duration_seconds)}</p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">Direction</p>
              <p className="font-medium capitalize">{call.direction}</p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">Status</p>
              <Badge
                variant={
                  call.status === "completed"
                    ? "success"
                    : call.status === "active"
                    ? "default"
                    : "destructive"
                }
              >
                {call.status}
              </Badge>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">Agent</p>
              <p className="font-medium">{call.agent_id || "Unknown"}</p>
            </div>
          </div>

          {/* Transcript */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="font-semibold">Transcript</h4>
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
            </div>
            <div className="rounded-lg border bg-muted/30 p-4 max-h-[300px] overflow-y-auto space-y-4">
              {transcript?.messages?.length > 0 ? (
                transcript.messages.map((msg: { role: string; content: string; timestamp: string }, i: number) => (
                  <div
                    key={i}
                    className={cn(
                      "flex gap-3",
                      msg.role === "assistant" ? "flex-row" : "flex-row-reverse"
                    )}
                  >
                    <div
                      className={cn(
                        "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
                        msg.role === "assistant" ? "bg-primary/10" : "bg-secondary"
                      )}
                    >
                      {msg.role === "assistant" ? (
                        <Bot className="h-4 w-4 text-primary" />
                      ) : (
                        <User className="h-4 w-4" />
                      )}
                    </div>
                    <div
                      className={cn(
                        "rounded-lg px-3 py-2 max-w-[80%]",
                        msg.role === "assistant"
                          ? "bg-primary/10 text-foreground"
                          : "bg-secondary"
                      )}
                    >
                      <p className="text-sm">{msg.content}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {new Date(msg.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-center text-muted-foreground py-8">
                  No transcript available
                </p>
              )}
            </div>
          </div>

          {/* Recording */}
          <div className="space-y-3">
            <h4 className="font-semibold">Recording</h4>
            <div className="flex items-center gap-4 rounded-lg border p-4">
              <Button variant="outline" size="icon">
                <Play className="h-4 w-4" />
              </Button>
              <div className="flex-1">
                <div className="h-2 bg-muted rounded-full">
                  <div className="h-2 bg-primary rounded-full w-0" />
                </div>
              </div>
              <span className="text-sm text-muted-foreground">
                {formatDuration(call.duration_seconds)}
              </span>
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default function CallsPage() {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [directionFilter, setDirectionFilter] = useState<string>("all");
  const [selectedCall, setSelectedCall] = useState<Call | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: ["calls", statusFilter, directionFilter],
    queryFn: () =>
      callsApi.list({
        status: statusFilter === "all" ? undefined : statusFilter,
      }),
  });

  const { data: stats } = useQuery({
    queryKey: ["call-stats"],
    queryFn: () => callsApi.getStats(),
  });

  const calls = data?.calls || [];
  const filteredCalls = calls.filter((call: Call) => {
    const matchesSearch = call.phone_number.includes(search);
    const matchesDirection =
      directionFilter === "all" || call.direction === directionFilter;
    return matchesSearch && matchesDirection;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Calls</h1>
          <p className="text-muted-foreground">
            Monitor and review all voice interactions
          </p>
        </div>
        <Button variant="outline">
          <Download className="mr-2 h-4 w-4" />
          Export
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Calls</CardTitle>
            <PhoneCall className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.total_calls || 0}</div>
            <p className="text-xs text-muted-foreground">All time</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Today</CardTitle>
            <Calendar className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.calls_today || 0}</div>
            <p className="text-xs text-muted-foreground">+12% from yesterday</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Duration</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatDuration(stats?.avg_duration || 0)}
            </div>
            <p className="text-xs text-muted-foreground">Per call</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.success_rate || 98}%</div>
            <p className="text-xs text-muted-foreground">Completed calls</p>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search by phone number..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-10"
          />
        </div>
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-[150px]">
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="completed">Completed</SelectItem>
            <SelectItem value="failed">Failed</SelectItem>
          </SelectContent>
        </Select>
        <Select value={directionFilter} onValueChange={setDirectionFilter}>
          <SelectTrigger className="w-[150px]">
            <SelectValue placeholder="Direction" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Directions</SelectItem>
            <SelectItem value="inbound">Inbound</SelectItem>
            <SelectItem value="outbound">Outbound</SelectItem>
          </SelectContent>
        </Select>
        <Button variant="outline" size="icon">
          <Filter className="h-4 w-4" />
        </Button>
      </div>

      {/* Calls Table */}
      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Phone Number</TableHead>
                <TableHead>Direction</TableHead>
                <TableHead>Agent</TableHead>
                <TableHead>Duration</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Time</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                [...Array(5)].map((_, i) => (
                  <TableRow key={i}>
                    <TableCell colSpan={7}>
                      <div className="h-12 animate-pulse bg-muted rounded" />
                    </TableCell>
                  </TableRow>
                ))
              ) : filteredCalls.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-12">
                    <PhoneCall className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-lg font-medium">No calls found</p>
                    <p className="text-muted-foreground">
                      {search ? "Try a different search" : "Calls will appear here once agents start taking calls"}
                    </p>
                  </TableCell>
                </TableRow>
              ) : (
                filteredCalls.map((call: Call) => (
                  <TableRow
                    key={call.id}
                    className="cursor-pointer hover:bg-muted/50"
                    onClick={() => {
                      setSelectedCall(call);
                      setDetailsOpen(true);
                    }}
                  >
                    <TableCell className="font-medium">
                      {formatPhoneNumber(call.phone_number)}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        {call.direction === "inbound" ? (
                          <PhoneIncoming className="h-4 w-4 text-success" />
                        ) : (
                          <PhoneOutgoing className="h-4 w-4 text-primary" />
                        )}
                        <span className="capitalize">{call.direction}</span>
                      </div>
                    </TableCell>
                    <TableCell>{call.agent_id || "—"}</TableCell>
                    <TableCell>{formatDuration(call.duration_seconds)}</TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          call.status === "completed"
                            ? "success"
                            : call.status === "active"
                            ? "default"
                            : "destructive"
                        }
                      >
                        {call.status === "active" && (
                          <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
                        )}
                        {call.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {formatRelativeTime(call.started_at)}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button variant="ghost" size="sm">
                        View
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Call Details Dialog */}
      <CallDetailsDialog
        call={selectedCall}
        open={detailsOpen}
        onOpenChange={setDetailsOpen}
      />
    </div>
  );
}

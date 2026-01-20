"use client";

import React, { useState, useEffect, useCallback } from "react";
import { DashboardLayout } from "@/components/layouts/dashboard-layout";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Button,
  Badge,
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Progress,
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
  Skeleton,
} from "@/components/ui";
import {
  Phone,
  PhoneCall,
  PhoneOff,
  Users,
  Clock,
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Pause,
  Play,
  MoreVertical,
  RefreshCw,
  Filter,
  Search,
  Maximize2,
  Minimize2,
  Radio,
  Eye,
  EyeOff,
  MessageSquare,
  Bot,
  User,
  Zap,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  CheckCircle,
  XCircle,
  AlertCircle,
  Signal,
  Wifi,
  WifiOff,
} from "lucide-react";
import { cn } from "@/lib/utils";

// Types
interface LiveCall {
  id: string;
  agent_id: string;
  agent_name: string;
  caller_number: string;
  caller_name?: string;
  direction: "inbound" | "outbound";
  status: "ringing" | "active" | "on_hold" | "transferring";
  started_at: string;
  duration: number;
  sentiment: "positive" | "neutral" | "negative";
  is_monitored: boolean;
  transcript_preview?: string;
  audio_level: number;
  connection_quality: "excellent" | "good" | "fair" | "poor";
}

interface LiveMetrics {
  active_calls: number;
  calls_in_queue: number;
  avg_wait_time: number;
  avg_handle_time: number;
  total_calls_today: number;
  calls_per_hour: number;
  success_rate: number;
  sentiment_breakdown: {
    positive: number;
    neutral: number;
    negative: number;
  };
}

interface AgentStatus {
  id: string;
  name: string;
  status: "active" | "busy" | "idle" | "offline";
  current_call?: string;
  calls_handled: number;
  avg_handle_time: number;
  success_rate: number;
}

// Mock data generators
const generateMockCalls = (): LiveCall[] => [
  {
    id: "call-1",
    agent_id: "agent-1",
    agent_name: "Sales Assistant",
    caller_number: "+1 (555) 123-4567",
    caller_name: "John Smith",
    direction: "inbound",
    status: "active",
    started_at: new Date(Date.now() - 180000).toISOString(),
    duration: 180,
    sentiment: "positive",
    is_monitored: false,
    transcript_preview: "I'm interested in upgrading my subscription...",
    audio_level: 0.7,
    connection_quality: "excellent",
  },
  {
    id: "call-2",
    agent_id: "agent-2",
    agent_name: "Support Agent",
    caller_number: "+1 (555) 987-6543",
    caller_name: "Sarah Johnson",
    direction: "inbound",
    status: "active",
    started_at: new Date(Date.now() - 420000).toISOString(),
    duration: 420,
    sentiment: "neutral",
    is_monitored: true,
    transcript_preview: "Can you help me with my billing issue?",
    audio_level: 0.5,
    connection_quality: "good",
  },
  {
    id: "call-3",
    agent_id: "agent-1",
    agent_name: "Sales Assistant",
    caller_number: "+1 (555) 456-7890",
    direction: "outbound",
    status: "ringing",
    started_at: new Date(Date.now() - 15000).toISOString(),
    duration: 15,
    sentiment: "neutral",
    is_monitored: false,
    audio_level: 0,
    connection_quality: "excellent",
  },
  {
    id: "call-4",
    agent_id: "agent-3",
    agent_name: "Appointment Scheduler",
    caller_number: "+1 (555) 321-0987",
    caller_name: "Mike Davis",
    direction: "inbound",
    status: "on_hold",
    started_at: new Date(Date.now() - 300000).toISOString(),
    duration: 300,
    sentiment: "negative",
    is_monitored: false,
    transcript_preview: "I've been waiting for a while now...",
    audio_level: 0,
    connection_quality: "fair",
  },
];

const generateMockMetrics = (): LiveMetrics => ({
  active_calls: 4,
  calls_in_queue: 2,
  avg_wait_time: 45,
  avg_handle_time: 285,
  total_calls_today: 156,
  calls_per_hour: 12,
  success_rate: 87.5,
  sentiment_breakdown: {
    positive: 45,
    neutral: 38,
    negative: 17,
  },
});

const generateMockAgents = (): AgentStatus[] => [
  {
    id: "agent-1",
    name: "Sales Assistant",
    status: "busy",
    current_call: "call-1",
    calls_handled: 28,
    avg_handle_time: 245,
    success_rate: 92,
  },
  {
    id: "agent-2",
    name: "Support Agent",
    status: "busy",
    current_call: "call-2",
    calls_handled: 34,
    avg_handle_time: 312,
    success_rate: 88,
  },
  {
    id: "agent-3",
    name: "Appointment Scheduler",
    status: "busy",
    current_call: "call-4",
    calls_handled: 42,
    avg_handle_time: 198,
    success_rate: 95,
  },
  {
    id: "agent-4",
    name: "Lead Qualifier",
    status: "idle",
    calls_handled: 18,
    avg_handle_time: 156,
    success_rate: 78,
  },
  {
    id: "agent-5",
    name: "Customer Service",
    status: "offline",
    calls_handled: 0,
    avg_handle_time: 0,
    success_rate: 0,
  },
];

// Utility functions
const formatDuration = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
};

const formatTime = (date: string): string => {
  return new Date(date).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });
};

// Components
function MetricCard({
  title,
  value,
  change,
  changeType,
  icon: Icon,
  trend,
}: {
  title: string;
  value: string | number;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon: React.ElementType;
  trend?: "up" | "down";
}) {
  return (
    <Card className="relative overflow-hidden">
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold">{value}</p>
            {change && (
              <div className="flex items-center gap-1 text-xs">
                {trend === "up" ? (
                  <ArrowUpRight
                    className={cn(
                      "h-3 w-3",
                      changeType === "positive" ? "text-green-500" : "text-red-500"
                    )}
                  />
                ) : trend === "down" ? (
                  <ArrowDownRight
                    className={cn(
                      "h-3 w-3",
                      changeType === "positive" ? "text-green-500" : "text-red-500"
                    )}
                  />
                ) : null}
                <span
                  className={cn(
                    changeType === "positive" && "text-green-500",
                    changeType === "negative" && "text-red-500",
                    changeType === "neutral" && "text-muted-foreground"
                  )}
                >
                  {change}
                </span>
              </div>
            )}
          </div>
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
            <Icon className="h-6 w-6 text-primary" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function LiveCallCard({
  call,
  onMonitor,
  onEndCall,
  isExpanded,
  onToggleExpand,
}: {
  call: LiveCall;
  onMonitor: () => void;
  onEndCall: () => void;
  isExpanded: boolean;
  onToggleExpand: () => void;
}) {
  const [duration, setDuration] = useState(call.duration);

  useEffect(() => {
    if (call.status === "active") {
      const interval = setInterval(() => {
        setDuration((d) => d + 1);
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [call.status]);

  const statusColors = {
    ringing: "bg-yellow-500",
    active: "bg-green-500",
    on_hold: "bg-orange-500",
    transferring: "bg-blue-500",
  };

  const sentimentColors = {
    positive: "text-green-500",
    neutral: "text-gray-500",
    negative: "text-red-500",
  };

  const qualityIcons = {
    excellent: <Signal className="h-4 w-4 text-green-500" />,
    good: <Signal className="h-4 w-4 text-green-400" />,
    fair: <Signal className="h-4 w-4 text-yellow-500" />,
    poor: <Signal className="h-4 w-4 text-red-500" />,
  };

  return (
    <Card
      className={cn(
        "transition-all duration-200",
        isExpanded && "ring-2 ring-primary",
        call.is_monitored && "border-blue-500 border-2"
      )}
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3">
            <div className="relative">
              <div
                className={cn(
                  "flex h-10 w-10 items-center justify-center rounded-full",
                  call.direction === "inbound" ? "bg-green-100" : "bg-blue-100"
                )}
              >
                {call.direction === "inbound" ? (
                  <PhoneCall className="h-5 w-5 text-green-600" />
                ) : (
                  <Phone className="h-5 w-5 text-blue-600" />
                )}
              </div>
              <span
                className={cn(
                  "absolute -bottom-1 -right-1 h-3 w-3 rounded-full border-2 border-white",
                  statusColors[call.status]
                )}
              />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-medium">
                  {call.caller_name || call.caller_number}
                </span>
                {call.is_monitored && (
                  <Badge variant="outline" className="text-xs gap-1">
                    <Eye className="h-3 w-3" />
                    Monitoring
                  </Badge>
                )}
              </div>
              <p className="text-sm text-muted-foreground">{call.caller_number}</p>
              <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                <Bot className="h-3 w-3" />
                <span>{call.agent_name}</span>
                <span>•</span>
                <span className="capitalize">{call.status}</span>
              </div>
            </div>
          </div>
          <div className="flex flex-col items-end gap-2">
            <div className="flex items-center gap-2">
              {qualityIcons[call.connection_quality]}
              <span className="font-mono text-lg font-semibold">
                {formatDuration(duration)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      onClick={onMonitor}
                      className={cn(call.is_monitored && "text-blue-500")}
                    >
                      {call.is_monitored ? (
                        <Eye className="h-4 w-4" />
                      ) : (
                        <EyeOff className="h-4 w-4" />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {call.is_monitored ? "Stop Monitoring" : "Start Monitoring"}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon-sm" onClick={onToggleExpand}>
                      {isExpanded ? (
                        <Minimize2 className="h-4 w-4" />
                      ) : (
                        <Maximize2 className="h-4 w-4" />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {isExpanded ? "Collapse" : "Expand"}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      onClick={onEndCall}
                      className="text-red-500 hover:text-red-600 hover:bg-red-50"
                    >
                      <PhoneOff className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>End Call</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
        </div>

        {/* Audio level indicator */}
        {call.status === "active" && (
          <div className="mt-3 flex items-center gap-2">
            <Volume2 className="h-4 w-4 text-muted-foreground" />
            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500 transition-all duration-100 rounded-full"
                style={{ width: `${call.audio_level * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Expanded content */}
        {isExpanded && (
          <div className="mt-4 pt-4 border-t space-y-4">
            {/* Sentiment indicator */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Sentiment</span>
              <div className="flex items-center gap-2">
                <span className={cn("text-sm font-medium capitalize", sentimentColors[call.sentiment])}>
                  {call.sentiment}
                </span>
                {call.sentiment === "positive" && <TrendingUp className="h-4 w-4 text-green-500" />}
                {call.sentiment === "negative" && <TrendingDown className="h-4 w-4 text-red-500" />}
              </div>
            </div>

            {/* Transcript preview */}
            {call.transcript_preview && (
              <div className="space-y-2">
                <span className="text-sm text-muted-foreground">Latest Message</span>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <div className="flex items-start gap-2">
                    <User className="h-4 w-4 mt-0.5 text-muted-foreground" />
                    <p className="text-sm">{call.transcript_preview}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Quick actions */}
            <div className="flex gap-2">
              <Button variant="outline" size="sm" className="flex-1">
                <MessageSquare className="h-4 w-4 mr-2" />
                View Full Transcript
              </Button>
              <Button variant="outline" size="sm" className="flex-1">
                <Zap className="h-4 w-4 mr-2" />
                Whisper to Agent
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function AgentStatusCard({ agent }: { agent: AgentStatus }) {
  const statusColors = {
    active: "bg-green-500",
    busy: "bg-yellow-500",
    idle: "bg-gray-400",
    offline: "bg-red-500",
  };

  const statusLabels = {
    active: "Active",
    busy: "On Call",
    idle: "Idle",
    offline: "Offline",
  };

  return (
    <div className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors">
      <div className="flex items-center gap-3">
        <div className="relative">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10">
            <Bot className="h-5 w-5 text-primary" />
          </div>
          <span
            className={cn(
              "absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 border-white",
              statusColors[agent.status]
            )}
          />
        </div>
        <div>
          <p className="font-medium text-sm">{agent.name}</p>
          <p className="text-xs text-muted-foreground">{statusLabels[agent.status]}</p>
        </div>
      </div>
      <div className="text-right">
        <p className="text-sm font-medium">{agent.calls_handled} calls</p>
        <p className="text-xs text-muted-foreground">
          {agent.success_rate > 0 ? `${agent.success_rate}% success` : "-"}
        </p>
      </div>
    </div>
  );
}

function SentimentGauge({
  breakdown,
}: {
  breakdown: LiveMetrics["sentiment_breakdown"];
}) {
  const total = breakdown.positive + breakdown.neutral + breakdown.negative;
  const positivePercent = (breakdown.positive / total) * 100;
  const neutralPercent = (breakdown.neutral / total) * 100;
  const negativePercent = (breakdown.negative / total) * 100;

  return (
    <div className="space-y-3">
      <div className="flex h-3 w-full overflow-hidden rounded-full">
        <div
          className="bg-green-500 transition-all"
          style={{ width: `${positivePercent}%` }}
        />
        <div
          className="bg-gray-400 transition-all"
          style={{ width: `${neutralPercent}%` }}
        />
        <div
          className="bg-red-500 transition-all"
          style={{ width: `${negativePercent}%` }}
        />
      </div>
      <div className="flex justify-between text-xs">
        <div className="flex items-center gap-1">
          <div className="h-2 w-2 rounded-full bg-green-500" />
          <span>Positive ({breakdown.positive}%)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="h-2 w-2 rounded-full bg-gray-400" />
          <span>Neutral ({breakdown.neutral}%)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="h-2 w-2 rounded-full bg-red-500" />
          <span>Negative ({breakdown.negative}%)</span>
        </div>
      </div>
    </div>
  );
}

export default function LiveCallsPage() {
  const [calls, setCalls] = useState<LiveCall[]>([]);
  const [metrics, setMetrics] = useState<LiveMetrics | null>(null);
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [expandedCall, setExpandedCall] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [isConnected, setIsConnected] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  // Simulate real-time updates
  useEffect(() => {
    // Initial load
    const loadData = () => {
      setCalls(generateMockCalls());
      setMetrics(generateMockMetrics());
      setAgents(generateMockAgents());
      setIsLoading(false);
      setLastUpdated(new Date());
    };

    loadData();

    // Simulate real-time updates every 3 seconds
    const interval = setInterval(() => {
      // Randomly update audio levels
      setCalls((prev) =>
        prev.map((call) => ({
          ...call,
          audio_level: call.status === "active" ? Math.random() * 0.8 + 0.2 : 0,
        }))
      );
      setLastUpdated(new Date());
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const handleMonitor = (callId: string) => {
    setCalls((prev) =>
      prev.map((call) =>
        call.id === callId ? { ...call, is_monitored: !call.is_monitored } : call
      )
    );
  };

  const handleEndCall = (callId: string) => {
    setCalls((prev) => prev.filter((call) => call.id !== callId));
  };

  const handleRefresh = () => {
    setIsLoading(true);
    setTimeout(() => {
      setCalls(generateMockCalls());
      setMetrics(generateMockMetrics());
      setAgents(generateMockAgents());
      setIsLoading(false);
      setLastUpdated(new Date());
    }, 500);
  };

  // Filter calls
  const filteredCalls = calls.filter((call) => {
    const matchesStatus = filterStatus === "all" || call.status === filterStatus;
    const matchesSearch =
      !searchQuery ||
      call.caller_number.includes(searchQuery) ||
      call.caller_name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      call.agent_name.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesStatus && matchesSearch;
  });

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Live Call Monitor</h1>
            <div className="flex items-center gap-2 mt-1">
              <div
                className={cn(
                  "flex items-center gap-1.5 text-sm",
                  isConnected ? "text-green-500" : "text-red-500"
                )}
              >
                {isConnected ? (
                  <>
                    <Radio className="h-3 w-3 animate-pulse" />
                    <span>Live</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="h-3 w-3" />
                    <span>Disconnected</span>
                  </>
                )}
              </div>
              <span className="text-muted-foreground text-sm">•</span>
              <span className="text-muted-foreground text-sm">
                Last updated: {lastUpdated.toLocaleTimeString()}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleRefresh}>
              <RefreshCw
                className={cn("h-4 w-4 mr-2", isLoading && "animate-spin")}
              />
              Refresh
            </Button>
            <Button size="sm">
              <Phone className="h-4 w-4 mr-2" />
              Start New Call
            </Button>
          </div>
        </div>

        {/* Real-time Metrics */}
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {isLoading ? (
            Array.from({ length: 4 }).map((_, i) => (
              <Card key={i}>
                <CardContent className="p-4">
                  <Skeleton className="h-20 w-full" />
                </CardContent>
              </Card>
            ))
          ) : metrics ? (
            <>
              <MetricCard
                title="Active Calls"
                value={metrics.active_calls}
                icon={PhoneCall}
                change="+2 from last hour"
                changeType="neutral"
              />
              <MetricCard
                title="Calls in Queue"
                value={metrics.calls_in_queue}
                icon={Users}
                change={metrics.calls_in_queue > 5 ? "High volume" : "Normal"}
                changeType={metrics.calls_in_queue > 5 ? "negative" : "positive"}
              />
              <MetricCard
                title="Avg Wait Time"
                value={`${metrics.avg_wait_time}s`}
                icon={Clock}
                change="-15s from avg"
                changeType="positive"
                trend="down"
              />
              <MetricCard
                title="Success Rate"
                value={`${metrics.success_rate}%`}
                icon={TrendingUp}
                change="+2.5% today"
                changeType="positive"
                trend="up"
              />
            </>
          ) : null}
        </div>

        {/* Main Content */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Active Calls List */}
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Active Calls</CardTitle>
                    <CardDescription>
                      {filteredCalls.length} call{filteredCalls.length !== 1 ? "s" : ""} in
                      progress
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="relative">
                      <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                      <Input
                        placeholder="Search calls..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-9 w-[200px]"
                      />
                    </div>
                    <Select value={filterStatus} onValueChange={setFilterStatus}>
                      <SelectTrigger className="w-[130px]">
                        <Filter className="h-4 w-4 mr-2" />
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Status</SelectItem>
                        <SelectItem value="active">Active</SelectItem>
                        <SelectItem value="ringing">Ringing</SelectItem>
                        <SelectItem value="on_hold">On Hold</SelectItem>
                        <SelectItem value="transferring">Transferring</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                {isLoading ? (
                  Array.from({ length: 3 }).map((_, i) => (
                    <Skeleton key={i} className="h-24 w-full" />
                  ))
                ) : filteredCalls.length > 0 ? (
                  filteredCalls.map((call) => (
                    <LiveCallCard
                      key={call.id}
                      call={call}
                      onMonitor={() => handleMonitor(call.id)}
                      onEndCall={() => handleEndCall(call.id)}
                      isExpanded={expandedCall === call.id}
                      onToggleExpand={() =>
                        setExpandedCall(expandedCall === call.id ? null : call.id)
                      }
                    />
                  ))
                ) : (
                  <div className="text-center py-12">
                    <Phone className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-medium">No Active Calls</h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      {searchQuery || filterStatus !== "all"
                        ? "No calls match your filters"
                        : "Calls will appear here when they start"}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            {/* Agent Status */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Agent Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {isLoading ? (
                  Array.from({ length: 4 }).map((_, i) => (
                    <Skeleton key={i} className="h-14 w-full" />
                  ))
                ) : (
                  agents.map((agent) => (
                    <AgentStatusCard key={agent.id} agent={agent} />
                  ))
                )}
              </CardContent>
            </Card>

            {/* Sentiment Overview */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Live Sentiment</CardTitle>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <Skeleton className="h-16 w-full" />
                ) : metrics ? (
                  <SentimentGauge breakdown={metrics.sentiment_breakdown} />
                ) : null}
              </CardContent>
            </Card>

            {/* Quick Stats */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Today's Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {isLoading ? (
                  <Skeleton className="h-24 w-full" />
                ) : metrics ? (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">
                        Total Calls
                      </span>
                      <span className="font-medium">{metrics.total_calls_today}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">
                        Calls/Hour
                      </span>
                      <span className="font-medium">{metrics.calls_per_hour}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">
                        Avg Handle Time
                      </span>
                      <span className="font-medium">
                        {formatDuration(metrics.avg_handle_time)}
                      </span>
                    </div>
                    <div className="pt-2 border-t">
                      <Button variant="outline" className="w-full" size="sm">
                        <BarChart3 className="h-4 w-4 mr-2" />
                        View Full Analytics
                      </Button>
                    </div>
                  </>
                ) : null}
              </CardContent>
            </Card>

            {/* Alerts */}
            <Card className="border-yellow-200 bg-yellow-50 dark:border-yellow-900 dark:bg-yellow-950/20">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-yellow-600" />
                  Active Alerts
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-start gap-2 text-sm">
                  <AlertCircle className="h-4 w-4 text-yellow-600 mt-0.5" />
                  <div>
                    <p className="font-medium">High wait time detected</p>
                    <p className="text-xs text-muted-foreground">
                      Queue wait time exceeds 2 minutes
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-2 text-sm">
                  <AlertCircle className="h-4 w-4 text-yellow-600 mt-0.5" />
                  <div>
                    <p className="font-medium">Negative sentiment spike</p>
                    <p className="text-xs text-muted-foreground">
                      Call #call-4 showing frustration
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

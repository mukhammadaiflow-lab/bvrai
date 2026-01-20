"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Bot,
  ArrowLeft,
  Play,
  Pause,
  Copy,
  Trash2,
  Settings,
  Phone,
  Clock,
  TrendingUp,
  TrendingDown,
  Edit2,
  Volume2,
  Brain,
  Wrench,
  ExternalLink,
  Calendar,
  BarChart3,
  MessageSquare,
  Loader2,
  AlertCircle,
  CheckCircle2,
  XCircle,
  MoreHorizontal,
  PhoneCall,
  PhoneOff,
  Download,
  RefreshCw,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress, CircularProgress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { StatCard, StatGrid } from "@/components/ui/stat-card";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import { AgentWizard } from "@/components/agents";
import { cn, formatRelativeTime, formatNumber, formatDuration, getStatusColor } from "@/lib/utils";
import { agents, calls } from "@/lib/api";
import { toast } from "sonner";

// Skeleton components for loading states
function AgentDetailSkeleton() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Skeleton className="h-8 w-8 rounded" />
        <Skeleton className="h-8 w-48" />
      </div>
      <div className="flex items-center gap-6">
        <Skeleton className="h-20 w-20 rounded-2xl" />
        <div className="space-y-2">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-5 w-96" />
          <div className="flex gap-2 pt-2">
            <Skeleton className="h-6 w-16 rounded-full" />
            <Skeleton className="h-6 w-24 rounded-full" />
            <Skeleton className="h-6 w-20 rounded-full" />
          </div>
        </div>
      </div>
      <div className="grid grid-cols-4 gap-4">
        <Skeleton className="h-28" />
        <Skeleton className="h-28" />
        <Skeleton className="h-28" />
        <Skeleton className="h-28" />
      </div>
    </div>
  );
}

function CallsListSkeleton() {
  return (
    <div className="space-y-3">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="flex items-center gap-4 p-4 border rounded-lg">
          <Skeleton className="h-10 w-10 rounded-full" />
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-48" />
          </div>
          <Skeleton className="h-6 w-16 rounded-full" />
          <Skeleton className="h-4 w-20" />
        </div>
      ))}
    </div>
  );
}

export default function AgentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const agentId = params.id as string;

  const [activeTab, setActiveTab] = useState("overview");
  const [isEditing, setIsEditing] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  // Fetch agent details
  const { data: agent, isLoading, error } = useQuery({
    queryKey: ["agent", agentId],
    queryFn: () => agents.get(agentId),
    enabled: !!agentId,
  });

  // Fetch agent calls
  const { data: agentCalls, isLoading: callsLoading } = useQuery({
    queryKey: ["agent-calls", agentId],
    queryFn: () => calls.list({ agent_id: agentId, page_size: 10 }),
    enabled: !!agentId,
  });

  // Toggle active mutation
  const toggleActiveMutation = useMutation({
    mutationFn: (isActive: boolean) => agents.update(agentId, { status: isActive ? 'active' : 'paused' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agent", agentId] });
      queryClient.invalidateQueries({ queryKey: ["agents"] });
      toast.success(agent?.status === 'active' ? "Agent paused" : "Agent activated");
    },
    onError: (error: Error) => {
      toast.error(`Failed to update agent: ${error.message}`);
    },
  });

  // Duplicate mutation
  const duplicateMutation = useMutation({
    mutationFn: () => agents.duplicate(agentId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["agents"] });
      toast.success("Agent duplicated successfully");
      router.push(`/agents/${data.id}`);
    },
    onError: (error: Error) => {
      toast.error(`Failed to duplicate agent: ${error.message}`);
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: () => agents.delete(agentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agents"] });
      toast.success("Agent deleted successfully");
      router.push("/agents");
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete agent: ${error.message}`);
    },
  });

  // Helper to get agent status
  const getAgentStatus = () => {
    return agent?.status || "draft";
  };

  // Error state
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <AlertCircle className="h-12 w-12 text-destructive" />
        <p className="text-lg font-medium">Failed to load agent</p>
        <p className="text-sm text-muted-foreground">
          {error instanceof Error ? error.message : "Agent not found"}
        </p>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/agents">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Agents
            </Link>
          </Button>
          <Button onClick={() => window.location.reload()}>Retry</Button>
        </div>
      </div>
    );
  }

  // Loading state
  if (isLoading) {
    return <AgentDetailSkeleton />;
  }

  // Edit mode
  if (isEditing && agent) {
    return (
      <div className="space-y-6">
        <Button
          variant="ghost"
          onClick={() => setIsEditing(false)}
          className="mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Agent
        </Button>
        <AgentWizard
          mode="edit"
          agentId={agentId}
          initialData={{
            name: agent.name,
            description: agent.description ?? undefined,
            system_prompt: agent.system_prompt ?? undefined,
            greeting_message: agent.greeting_message ?? undefined,
            voice_provider: agent.voice_config?.provider,
            voice_id: agent.voice_config?.voice_id,
            voice_name: agent.voice_config?.voice_name,
            language: agent.voice_config?.language,
            speaking_rate: agent.voice_config?.speaking_rate,
            pitch: agent.voice_config?.pitch,
            llm_provider: agent.llm_config?.provider,
            llm_model: agent.llm_config?.model,
            temperature: agent.llm_config?.temperature,
            max_tokens: agent.llm_config?.max_tokens,
            tools: agent.tools?.filter((t: any) => t.enabled).map((t: any) => t.type) || [],
            is_active: agent.status === 'active',
            allow_interruptions: agent.settings?.allow_interruptions,
            end_call_on_silence: agent.settings?.end_call_on_silence,
            silence_timeout: agent.settings?.silence_timeout,
          }}
        />
      </div>
    );
  }

  const status = getAgentStatus();
  const recentCalls = agentCalls?.items || [];

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Link href="/agents" className="hover:text-foreground transition-colors">
          Agents
        </Link>
        <span>/</span>
        <span className="text-foreground">{agent?.name}</span>
      </div>

      {/* Header */}
      <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
        <div className="flex items-start gap-6">
          <div
            className={cn(
              "flex h-20 w-20 items-center justify-center rounded-2xl",
              status === "active"
                ? "bg-green-100 dark:bg-green-950"
                : status === "paused"
                ? "bg-yellow-100 dark:bg-yellow-950"
                : "bg-gray-100 dark:bg-gray-900"
            )}
          >
            <Bot
              className={cn(
                "h-10 w-10",
                status === "active"
                  ? "text-green-600 dark:text-green-400"
                  : status === "paused"
                  ? "text-yellow-600 dark:text-yellow-400"
                  : "text-gray-600 dark:text-gray-400"
              )}
            />
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold">{agent?.name}</h1>
              <Badge className={getStatusColor(status)}>{status}</Badge>
            </div>
            <p className="text-muted-foreground mt-1 max-w-xl">
              {agent?.description || "No description provided"}
            </p>
            <div className="flex flex-wrap gap-2 mt-3">
              <Badge variant="secondary" className="font-normal">
                <Brain className="mr-1 h-3 w-3" />
                {agent?.llm_config?.model || "GPT-4o"}
              </Badge>
              <Badge variant="secondary" className="font-normal">
                <Volume2 className="mr-1 h-3 w-3" />
                {agent?.voice_config?.voice_name || "Default Voice"}
              </Badge>
              {(agent?.tools?.filter((t: any) => t.enabled).length ?? 0) > 0 && (
                <Badge variant="secondary" className="font-normal">
                  <Wrench className="mr-1 h-3 w-3" />
                  {agent?.tools?.filter((t: any) => t.enabled).length} tools
                </Badge>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant={status === "active" ? "outline" : "default"}
            onClick={() => toggleActiveMutation.mutate(agent?.status !== 'active')}
            disabled={toggleActiveMutation.isPending}
          >
            {toggleActiveMutation.isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : status === "active" ? (
              <Pause className="mr-2 h-4 w-4" />
            ) : (
              <Play className="mr-2 h-4 w-4" />
            )}
            {status === "active" ? "Pause" : "Activate"}
          </Button>
          <Button variant="outline" onClick={() => setIsEditing(true)}>
            <Edit2 className="mr-2 h-4 w-4" />
            Edit
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="icon">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem
                onClick={() => duplicateMutation.mutate()}
                disabled={duplicateMutation.isPending}
              >
                <Copy className="mr-2 h-4 w-4" />
                Duplicate
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Download className="mr-2 h-4 w-4" />
                Export Config
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive"
                onClick={() => setShowDeleteDialog(true)}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Stats */}
      <StatGrid columns={4}>
        <StatCard
          title="Total Calls"
          value={formatNumber(agent?.total_calls || 0)}
          icon={<Phone className="h-5 w-5" />}
          trend={agent?.calls_trend ? {
            value: agent.calls_trend,
            label: "vs last period"
          } : undefined}
        />
        <StatCard
          title="Success Rate"
          value={`${((agent?.success_rate || 0) * 100).toFixed(1)}%`}
          icon={<TrendingUp className="h-5 w-5" />}
          trend={agent?.success_trend ? {
            value: agent.success_trend,
            label: "vs last period"
          } : undefined}
        />
        <StatCard
          title="Avg Duration"
          value={formatDuration(agent?.avg_duration || 0)}
          icon={<Clock className="h-5 w-5" />}
        />
        <StatCard
          title="Active Sessions"
          value={agent?.active_sessions || 0}
          icon={<PhoneCall className="h-5 w-5" />}
          description={(agent?.active_sessions ?? 0) > 0 ? "Currently on call" : "No active calls"}
        />
      </StatGrid>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="calls">Call History</TabsTrigger>
          <TabsTrigger value="config">Configuration</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6 mt-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* System Prompt */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageSquare className="h-4 w-4" />
                  System Prompt
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="rounded-lg bg-muted/50 p-4 max-h-64 overflow-y-auto">
                  <p className="text-sm whitespace-pre-wrap font-mono">
                    {agent?.system_prompt || "No system prompt configured"}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Greeting Message */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Volume2 className="h-4 w-4" />
                  Greeting Message
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="rounded-lg bg-muted/50 p-4">
                  <p className="text-sm">
                    {agent?.greeting_message || "Hello! How can I help you today?"}
                  </p>
                </div>
                <Button variant="outline" size="sm" className="mt-4">
                  <Play className="mr-2 h-3 w-3" />
                  Preview Greeting
                </Button>
              </CardContent>
            </Card>

            {/* Tools */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Wrench className="h-4 w-4" />
                  Enabled Tools
                </CardTitle>
              </CardHeader>
              <CardContent>
                {(agent?.tools?.filter((t: any) => t.enabled).length ?? 0) > 0 ? (
                  <div className="space-y-2">
                    {agent?.tools
                      ?.filter((t: any) => t.enabled)
                      .map((tool: any) => (
                        <div
                          key={tool.type}
                          className="flex items-center gap-3 p-3 rounded-lg bg-muted/50"
                        >
                          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                            <Wrench className="h-4 w-4 text-primary" />
                          </div>
                          <div>
                            <p className="font-medium capitalize">{tool.type}</p>
                            <p className="text-xs text-muted-foreground">
                              {tool.description || "No description"}
                            </p>
                          </div>
                        </div>
                      ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No tools enabled. This agent responds conversationally without taking actions.
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Recent Activity */}
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-4 w-4" />
                  Recent Calls
                </CardTitle>
                <Button variant="ghost" size="sm" asChild>
                  <Link href={`/calls?agent=${agentId}`}>
                    View All
                    <ExternalLink className="ml-1 h-3 w-3" />
                  </Link>
                </Button>
              </CardHeader>
              <CardContent>
                {callsLoading ? (
                  <CallsListSkeleton />
                ) : recentCalls.length > 0 ? (
                  <div className="space-y-3">
                    {recentCalls.slice(0, 5).map((call: any) => (
                      <Link
                        key={call.id}
                        href={`/calls/${call.id}`}
                        className="flex items-center gap-3 p-3 rounded-lg hover:bg-muted/50 transition-colors"
                      >
                        <div
                          className={cn(
                            "flex h-10 w-10 items-center justify-center rounded-full",
                            call.status === "completed"
                              ? "bg-green-100 text-green-600"
                              : call.status === "failed"
                              ? "bg-red-100 text-red-600"
                              : "bg-yellow-100 text-yellow-600"
                          )}
                        >
                          {call.status === "completed" ? (
                            <CheckCircle2 className="h-5 w-5" />
                          ) : call.status === "failed" ? (
                            <XCircle className="h-5 w-5" />
                          ) : (
                            <PhoneOff className="h-5 w-5" />
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium truncate">
                            {call.caller_number || "Unknown Caller"}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {formatDuration(call.duration || 0)} duration
                          </p>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {formatRelativeTime(call.created_at)}
                        </span>
                      </Link>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-8">
                    No calls yet. Activate the agent to start receiving calls.
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Calls Tab */}
        <TabsContent value="calls" className="mt-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Call History</CardTitle>
                <CardDescription>
                  All calls handled by this agent
                </CardDescription>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm">
                  <Download className="mr-2 h-4 w-4" />
                  Export
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => queryClient.invalidateQueries({ queryKey: ["agent-calls", agentId] })}
                >
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {callsLoading ? (
                <CallsListSkeleton />
              ) : recentCalls.length > 0 ? (
                <div className="space-y-2">
                  {recentCalls.map((call: any) => (
                    <Link
                      key={call.id}
                      href={`/calls/${call.id}`}
                      className="flex items-center gap-4 p-4 rounded-lg border hover:bg-muted/50 transition-colors"
                    >
                      <div
                        className={cn(
                          "flex h-10 w-10 items-center justify-center rounded-full",
                          call.status === "completed"
                            ? "bg-green-100 text-green-600"
                            : call.status === "failed"
                            ? "bg-red-100 text-red-600"
                            : call.status === "in_progress"
                            ? "bg-blue-100 text-blue-600"
                            : "bg-yellow-100 text-yellow-600"
                        )}
                      >
                        {call.status === "completed" ? (
                          <CheckCircle2 className="h-5 w-5" />
                        ) : call.status === "failed" ? (
                          <XCircle className="h-5 w-5" />
                        ) : call.status === "in_progress" ? (
                          <PhoneCall className="h-5 w-5" />
                        ) : (
                          <PhoneOff className="h-5 w-5" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="font-medium">
                            {call.caller_number || "Unknown Caller"}
                          </p>
                          <Badge
                            variant={
                              call.direction === "inbound" ? "secondary" : "outline"
                            }
                            className="text-xs"
                          >
                            {call.direction}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground truncate">
                          {call.summary || "No summary available"}
                        </p>
                      </div>
                      <div className="text-right">
                        <Badge className={getStatusColor(call.status)}>
                          {call.status}
                        </Badge>
                        <p className="text-xs text-muted-foreground mt-1">
                          {formatDuration(call.duration || 0)}
                        </p>
                      </div>
                      <span className="text-sm text-muted-foreground whitespace-nowrap">
                        {formatRelativeTime(call.created_at)}
                      </span>
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Phone className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-lg font-medium">No calls yet</p>
                  <p className="text-sm text-muted-foreground">
                    {status === "active"
                      ? "Waiting for incoming calls"
                      : "Activate the agent to start receiving calls"}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Configuration Tab */}
        <TabsContent value="config" className="mt-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Voice Configuration */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Volume2 className="h-4 w-4" />
                  Voice Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Provider</p>
                    <p className="font-medium capitalize">
                      {agent?.voice_config?.provider || "ElevenLabs"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Voice</p>
                    <p className="font-medium">
                      {agent?.voice_config?.voice_name || "Default"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Language</p>
                    <p className="font-medium">
                      {agent?.voice_config?.language || "en-US"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Speaking Rate</p>
                    <p className="font-medium">
                      {agent?.voice_config?.speaking_rate || 1.0}x
                    </p>
                  </div>
                </div>
                <Button variant="outline" className="w-full" onClick={() => setIsEditing(true)}>
                  <Settings className="mr-2 h-4 w-4" />
                  Edit Voice Settings
                </Button>
              </CardContent>
            </Card>

            {/* AI Model Configuration */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-4 w-4" />
                  AI Model Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Model</p>
                    <p className="font-medium">{agent?.llm_config?.model || "GPT-4o"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Provider</p>
                    <p className="font-medium capitalize">
                      {agent?.llm_config?.provider || "OpenAI"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Temperature</p>
                    <p className="font-medium">{agent?.llm_config?.temperature || 0.7}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Max Tokens</p>
                    <p className="font-medium">{agent?.llm_config?.max_tokens || 1024}</p>
                  </div>
                </div>
                <Button variant="outline" className="w-full" onClick={() => setIsEditing(true)}>
                  <Settings className="mr-2 h-4 w-4" />
                  Edit AI Settings
                </Button>
              </CardContent>
            </Card>

            {/* Conversation Settings */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Conversation Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Allow Interruptions</p>
                    <p className="text-sm text-muted-foreground">
                      Let callers interrupt while agent speaks
                    </p>
                  </div>
                  <Badge variant={agent?.settings?.allow_interruptions ? "default" : "secondary"}>
                    {agent?.settings?.allow_interruptions ? "Enabled" : "Disabled"}
                  </Badge>
                </div>
                <Separator />
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">End Call on Silence</p>
                    <p className="text-sm text-muted-foreground">
                      Auto-end after {agent?.settings?.silence_timeout || 5}s silence
                    </p>
                  </div>
                  <Badge variant={agent?.settings?.end_call_on_silence ? "default" : "secondary"}>
                    {agent?.settings?.end_call_on_silence ? "Enabled" : "Disabled"}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Metadata */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  Metadata
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Created</p>
                    <p className="font-medium">{formatRelativeTime(agent?.created_at)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Last Updated</p>
                    <p className="font-medium">{formatRelativeTime(agent?.updated_at)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Agent ID</p>
                    <p className="font-mono text-xs truncate">{agent?.id}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Version</p>
                    <p className="font-medium">{agent?.version || 1}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="mt-6">
          <Alert className="mb-6">
            <BarChart3 className="h-4 w-4" />
            <AlertTitle>Analytics Dashboard</AlertTitle>
            <AlertDescription>
              Detailed analytics for this agent are available in the Analytics page.
            </AlertDescription>
          </Alert>

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Call Outcome Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Successful</span>
                      <span className="text-green-600">{((agent?.success_rate || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <Progress value={(agent?.success_rate || 0) * 100} className="h-2" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Failed</span>
                      <span className="text-red-600">{((1 - (agent?.success_rate || 0)) * 100).toFixed(0)}%</span>
                    </div>
                    <Progress value={(1 - (agent?.success_rate || 0)) * 100} className="h-2 [&>div]:bg-red-500" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Performance Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-center py-4">
                  <CircularProgress
                    value={(agent?.performance_score || 0.85) * 100}
                    size={160}
                    strokeWidth={12}
                    showLabel
                  />
                </div>
                <p className="text-center text-sm text-muted-foreground">
                  Based on success rate, call duration, and customer satisfaction
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="mt-6">
            <Button asChild>
              <Link href={`/analytics?agent=${agentId}`}>
                <BarChart3 className="mr-2 h-4 w-4" />
                View Full Analytics
              </Link>
            </Button>
          </div>
        </TabsContent>
      </Tabs>

      {/* Delete Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Agent</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete "{agent?.name}"? This action cannot be undone.
              All associated call history will be preserved but unlinked from this agent.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDeleteDialog(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                deleteMutation.mutate();
                setShowDeleteDialog(false);
              }}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              Delete Agent
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

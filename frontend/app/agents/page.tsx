"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Bot,
  Plus,
  Search,
  MoreHorizontal,
  Play,
  Pause,
  Copy,
  Trash2,
  Edit,
  Phone,
  Clock,
  TrendingUp,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn, formatRelativeTime, formatNumber, getStatusColor } from "@/lib/utils";
import { agents } from "@/lib/api";
import { toast } from "sonner";

// Skeleton component for loading states
function AgentCardSkeleton() {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="h-12 w-12 bg-muted animate-pulse rounded-lg" />
            <div>
              <div className="h-5 w-32 bg-muted animate-pulse rounded mb-2" />
              <div className="h-5 w-16 bg-muted animate-pulse rounded" />
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-4 w-full bg-muted animate-pulse rounded mb-2" />
        <div className="h-4 w-3/4 bg-muted animate-pulse rounded mb-4" />
        <div className="flex gap-2 mb-4">
          <div className="h-5 w-16 bg-muted animate-pulse rounded-full" />
          <div className="h-5 w-24 bg-muted animate-pulse rounded-full" />
        </div>
        <div className="h-20 bg-muted animate-pulse rounded mb-4" />
        <div className="flex gap-2">
          <div className="h-8 flex-1 bg-muted animate-pulse rounded" />
          <div className="h-8 flex-1 bg-muted animate-pulse rounded" />
        </div>
      </CardContent>
    </Card>
  );
}

export default function AgentsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Fetch agents from API
  const { data: agentsData, isLoading, error } = useQuery({
    queryKey: ["agents", "list"],
    queryFn: () => agents.list({ page_size: 50 }),
    staleTime: 30 * 1000, // 30 seconds
  });

  // Mutation for duplicating agents
  const duplicateMutation = useMutation({
    mutationFn: (agentId: string) => agents.duplicate(agentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agents"] });
      toast.success("Agent duplicated successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to duplicate agent: ${error.message}`);
    },
  });

  // Mutation for deleting agents
  const deleteMutation = useMutation({
    mutationFn: (agentId: string) => agents.delete(agentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agents"] });
      toast.success("Agent deleted successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete agent: ${error.message}`);
    },
  });

  const allAgents = agentsData?.items || [];

  const filteredAgents = allAgents.filter((agent: any) => {
    const matchesSearch =
      agent.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      agent.description?.toLowerCase().includes(searchQuery.toLowerCase());
    const agentStatus = agent.is_active ? "active" : agent.is_deployed ? "paused" : "draft";
    const matchesStatus = !statusFilter || agentStatus === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const statusCounts = {
    all: allAgents.length,
    active: allAgents.filter((a: any) => a.is_active).length,
    paused: allAgents.filter((a: any) => !a.is_active && a.is_deployed).length,
    draft: allAgents.filter((a: any) => !a.is_active && !a.is_deployed).length,
  };

  // Helper to get agent status
  const getAgentStatus = (agent: any) => {
    if (agent.is_active) return "active";
    if (agent.is_deployed) return "paused";
    return "draft";
  };

  // Error state
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <AlertCircle className="h-12 w-12 text-destructive" />
        <p className="text-lg font-medium">Failed to load agents</p>
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
          <h1 className="text-2xl font-bold">Voice Agents</h1>
          <p className="text-muted-foreground">
            Create and manage your AI voice agents
          </p>
        </div>
        <Button asChild>
          <Link href="/agents/new">
            <Plus className="mr-2 h-4 w-4" />
            Create Agent
          </Link>
        </Button>
      </div>

      {/* Filters */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center">
        <div className="flex-1">
          <Input
            placeholder="Search agents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            leftIcon={<Search className="h-4 w-4" />}
            className="max-w-md"
          />
        </div>
        <div className="flex gap-2">
          <Button
            variant={statusFilter === null ? "default" : "outline"}
            size="sm"
            onClick={() => setStatusFilter(null)}
          >
            All ({statusCounts.all})
          </Button>
          <Button
            variant={statusFilter === "active" ? "default" : "outline"}
            size="sm"
            onClick={() => setStatusFilter("active")}
          >
            Active ({statusCounts.active})
          </Button>
          <Button
            variant={statusFilter === "paused" ? "default" : "outline"}
            size="sm"
            onClick={() => setStatusFilter("paused")}
          >
            Paused ({statusCounts.paused})
          </Button>
          <Button
            variant={statusFilter === "draft" ? "default" : "outline"}
            size="sm"
            onClick={() => setStatusFilter("draft")}
          >
            Draft ({statusCounts.draft})
          </Button>
        </div>
      </div>

      {/* Loading State */}
      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <AgentCardSkeleton />
          <AgentCardSkeleton />
          <AgentCardSkeleton />
          <AgentCardSkeleton />
        </div>
      ) : (
        <>
          {/* Agents Grid */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {filteredAgents.map((agent: any) => {
              const status = getAgentStatus(agent);
              return (
                <Card key={agent.id} className="group relative overflow-hidden">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className={cn(
                          "flex h-12 w-12 items-center justify-center rounded-lg",
                          status === "active" ? "bg-green-100" :
                          status === "paused" ? "bg-yellow-100" :
                          "bg-gray-100"
                        )}>
                          <Bot className={cn(
                            "h-6 w-6",
                            status === "active" ? "text-green-600" :
                            status === "paused" ? "text-yellow-600" :
                            "text-gray-600"
                          )} />
                        </div>
                        <div>
                          <CardTitle className="text-lg">{agent.name}</CardTitle>
                          <Badge className={getStatusColor(status)}>
                            {status}
                          </Badge>
                        </div>
                      </div>
                      <Button variant="ghost" size="icon-sm">
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                      {agent.description || "No description"}
                    </p>

                    {/* Config Pills */}
                    <div className="flex flex-wrap gap-2 mb-4">
                      <span className="inline-flex items-center rounded-full bg-secondary px-2.5 py-0.5 text-xs">
                        {agent.llm_config?.model || "GPT-4o"}
                      </span>
                      <span className="inline-flex items-center rounded-full bg-secondary px-2.5 py-0.5 text-xs">
                        {agent.voice_config?.voice_name || "Default Voice"}
                      </span>
                    </div>

                    {/* Stats */}
                    {status !== "draft" && (
                      <div className="grid grid-cols-3 gap-2 rounded-lg bg-secondary/50 p-3">
                        <div className="text-center">
                          <div className="flex items-center justify-center gap-1 text-muted-foreground">
                            <Phone className="h-3 w-3" />
                          </div>
                          <p className="text-lg font-bold">{formatNumber(agent.total_calls || 0)}</p>
                          <p className="text-xs text-muted-foreground">calls</p>
                        </div>
                        <div className="text-center">
                          <div className="flex items-center justify-center gap-1 text-muted-foreground">
                            <TrendingUp className="h-3 w-3" />
                          </div>
                          <p className="text-lg font-bold text-green-600">
                            {((agent.success_rate || 0) * 100).toFixed(0)}%
                          </p>
                          <p className="text-xs text-muted-foreground">success</p>
                        </div>
                        <div className="text-center">
                          <div className="flex items-center justify-center gap-1 text-muted-foreground">
                            <Clock className="h-3 w-3" />
                          </div>
                          <p className="text-lg font-bold">{Math.round((agent.avg_duration || 0) / 60)}m</p>
                          <p className="text-xs text-muted-foreground">avg</p>
                        </div>
                      </div>
                    )}

                    {/* Actions */}
                    <div className="flex gap-2 mt-4">
                      <Button variant="outline" size="sm" className="flex-1" asChild>
                        <Link href={`/agents/${agent.id}`}>
                          <Edit className="mr-1 h-3 w-3" />
                          Edit
                        </Link>
                      </Button>
                      {status === "active" ? (
                        <Button variant="outline" size="sm" className="flex-1">
                          <Pause className="mr-1 h-3 w-3" />
                          Pause
                        </Button>
                      ) : (
                        <Button variant="outline" size="sm" className="flex-1">
                          <Play className="mr-1 h-3 w-3" />
                          Activate
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        onClick={() => duplicateMutation.mutate(agent.id)}
                        disabled={duplicateMutation.isPending}
                      >
                        {duplicateMutation.isPending ? (
                          <Loader2 className="h-3 w-3 animate-spin" />
                        ) : (
                          <Copy className="h-3 w-3" />
                        )}
                      </Button>
                    </div>

                    {/* Footer */}
                    <p className="text-xs text-muted-foreground mt-3">
                      Updated {formatRelativeTime(agent.updated_at || agent.created_at)}
                    </p>
                  </CardContent>
                </Card>
              );
            })}

            {/* Create New Card */}
            <Link href="/agents/new">
              <Card className="flex items-center justify-center border-dashed hover:border-primary hover:bg-primary/5 cursor-pointer transition-colors min-h-[300px]">
                <div className="text-center">
                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 mb-3">
                    <Plus className="h-6 w-6 text-primary" />
                  </div>
                  <p className="font-medium">Create New Agent</p>
                  <p className="text-sm text-muted-foreground">
                    Build a custom voice agent
                  </p>
                </div>
              </Card>
            </Link>
          </div>

          {/* Empty State */}
          {filteredAgents.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center rounded-lg border border-dashed p-12">
              <Bot className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No agents found</h3>
              <p className="text-muted-foreground mb-4">
                {searchQuery
                  ? "Try adjusting your search or filters"
                  : "Get started by creating your first agent"}
              </p>
              <Button asChild>
                <Link href="/agents/new">
                  <Plus className="mr-2 h-4 w-4" />
                  Create Agent
                </Link>
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

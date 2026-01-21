"use client";

import React, { useState, useEffect } from "react";
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
  Sparkles,
  Zap,
  Settings2,
  Filter,
  LayoutGrid,
  List,
  ChevronDown,
  Activity,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn, formatRelativeTime, formatNumber, getStatusColor } from "@/lib/utils";
import { agents } from "@/lib/api";
import { toast } from "sonner";

// Modern skeleton component for loading states
function AgentCardSkeleton() {
  return (
    <Card className="overflow-hidden border-0 shadow-lg shadow-primary/5 bg-gradient-to-br from-card to-card/80">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-4">
            <div className="h-14 w-14 bg-gradient-to-br from-muted to-muted/50 animate-pulse rounded-2xl" />
            <div className="space-y-2">
              <div className="h-5 w-36 bg-muted animate-pulse rounded-lg" />
              <div className="h-5 w-16 bg-muted animate-pulse rounded-full" />
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="h-4 w-full bg-muted animate-pulse rounded" />
          <div className="h-4 w-3/4 bg-muted animate-pulse rounded" />
        </div>
        <div className="flex gap-2">
          <div className="h-6 w-20 bg-muted animate-pulse rounded-full" />
          <div className="h-6 w-24 bg-muted animate-pulse rounded-full" />
        </div>
        <div className="h-24 bg-muted/50 animate-pulse rounded-xl" />
        <div className="flex gap-2">
          <div className="h-10 flex-1 bg-muted animate-pulse rounded-xl" />
          <div className="h-10 flex-1 bg-muted animate-pulse rounded-xl" />
        </div>
      </CardContent>
    </Card>
  );
}

export default function AgentsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [mounted, setMounted] = useState(false);
  const queryClient = useQueryClient();

  useEffect(() => {
    setMounted(true);
  }, []);

  // Fetch agents from API
  const { data: agentsData, isLoading, error } = useQuery({
    queryKey: ["agents", "list"],
    queryFn: () => agents.list({ page_size: 50 }),
    staleTime: 30 * 1000,
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

  const getAgentStatus = (agent: any) => {
    if (agent.is_active) return "active";
    if (agent.is_deployed) return "paused";
    return "draft";
  };

  // Error state
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] space-y-6">
        <div className="relative">
          <div className="absolute -inset-4 bg-destructive/10 rounded-full blur-2xl" />
          <div className="relative w-20 h-20 rounded-2xl bg-gradient-to-br from-destructive/20 to-destructive/10 flex items-center justify-center">
            <AlertCircle className="h-10 w-10 text-destructive" />
          </div>
        </div>
        <div className="text-center space-y-2">
          <h3 className="text-xl font-semibold">Failed to load agents</h3>
          <p className="text-muted-foreground max-w-md">
            {error instanceof Error ? error.message : "An unexpected error occurred"}
          </p>
        </div>
        <Button onClick={() => window.location.reload()} className="rounded-xl">
          Try Again
        </Button>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "space-y-8 transition-all duration-700",
        mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
      )}
    >
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-primary via-purple-600 to-accent p-8 text-white">
        {/* Background Effects */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-white/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-accent/20 rounded-full blur-3xl translate-y-1/2 -translate-x-1/2" />
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] bg-[size:32px_32px]" />

        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/20 backdrop-blur-sm text-sm font-medium">
              <Sparkles className="h-4 w-4" />
              <span>Voice AI Agents</span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
              Your Voice Agents
            </h1>
            <p className="text-white/70 text-lg max-w-xl">
              Create, manage, and deploy intelligent voice agents that handle calls naturally and efficiently.
            </p>
          </div>
          <div className="flex flex-col sm:flex-row gap-3">
            <Button
              asChild
              size="lg"
              className="bg-white text-primary hover:bg-white/90 shadow-xl shadow-black/20 rounded-xl font-semibold"
            >
              <Link href="/agents/new">
                <Plus className="mr-2 h-5 w-5" />
                Create Agent
              </Link>
            </Button>
            <Button
              asChild
              variant="outline"
              size="lg"
              className="border-white/30 bg-white/10 text-white hover:bg-white/20 backdrop-blur-sm rounded-xl"
            >
              <Link href="/templates">
                <Sparkles className="mr-2 h-5 w-5" />
                Templates
              </Link>
            </Button>
          </div>
        </div>

        {/* Stats */}
        <div className="relative z-10 grid grid-cols-2 md:grid-cols-4 gap-4 mt-8 pt-6 border-t border-white/20">
          <div className="text-center">
            <div className="text-3xl font-bold">{statusCounts.all}</div>
            <div className="text-white/60 text-sm">Total Agents</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-300">{statusCounts.active}</div>
            <div className="text-white/60 text-sm">Active</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-yellow-300">{statusCounts.paused}</div>
            <div className="text-white/60 text-sm">Paused</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-white/70">{statusCounts.draft}</div>
            <div className="text-white/60 text-sm">Draft</div>
          </div>
        </div>
      </div>

      {/* Filters Bar */}
      <div className="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
        <div className="flex flex-col sm:flex-row gap-3 w-full lg:w-auto">
          {/* Search */}
          <div className="relative flex-1 sm:w-80">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
            <Input
              placeholder="Search agents by name or description..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-11 h-12 rounded-xl border-2 border-border/50 bg-muted/30 focus:border-primary focus:bg-background transition-all"
            />
          </div>

          {/* Status Filter Pills */}
          <div className="flex gap-2 overflow-x-auto pb-2 sm:pb-0 scrollbar-hide">
            {[
              { key: null, label: "All", count: statusCounts.all },
              { key: "active", label: "Active", count: statusCounts.active, color: "bg-green-500" },
              { key: "paused", label: "Paused", count: statusCounts.paused, color: "bg-yellow-500" },
              { key: "draft", label: "Draft", count: statusCounts.draft, color: "bg-gray-400" },
            ].map((filter) => (
              <button
                key={filter.key || "all"}
                onClick={() => setStatusFilter(filter.key)}
                className={cn(
                  "flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium text-sm transition-all whitespace-nowrap",
                  statusFilter === filter.key
                    ? "bg-primary text-white shadow-lg shadow-primary/25"
                    : "bg-muted/50 text-muted-foreground hover:bg-muted hover:text-foreground"
                )}
              >
                {filter.color && (
                  <span className={cn("w-2 h-2 rounded-full", filter.color)} />
                )}
                {filter.label}
                <span className={cn(
                  "text-xs px-1.5 py-0.5 rounded-full",
                  statusFilter === filter.key ? "bg-white/20" : "bg-background"
                )}>
                  {filter.count}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-2 bg-muted/50 p-1 rounded-xl">
          <button
            onClick={() => setViewMode("grid")}
            className={cn(
              "p-2.5 rounded-lg transition-all",
              viewMode === "grid"
                ? "bg-background shadow-sm text-primary"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <LayoutGrid className="h-5 w-5" />
          </button>
          <button
            onClick={() => setViewMode("list")}
            className={cn(
              "p-2.5 rounded-lg transition-all",
              viewMode === "list"
                ? "bg-background shadow-sm text-primary"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <List className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Loading State */}
      {isLoading ? (
        <div className={cn(
          "grid gap-6",
          viewMode === "grid" ? "md:grid-cols-2 lg:grid-cols-3" : "grid-cols-1"
        )}>
          <AgentCardSkeleton />
          <AgentCardSkeleton />
          <AgentCardSkeleton />
          <AgentCardSkeleton />
        </div>
      ) : (
        <>
          {/* Agents Grid/List */}
          <div className={cn(
            "grid gap-6",
            viewMode === "grid" ? "md:grid-cols-2 lg:grid-cols-3" : "grid-cols-1"
          )}>
            {filteredAgents.map((agent: any, index: number) => {
              const status = getAgentStatus(agent);
              return (
                <Card
                  key={agent.id}
                  className={cn(
                    "group relative overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300",
                    status === "active" && "shadow-green-500/10 hover:shadow-green-500/20",
                    status === "paused" && "shadow-yellow-500/10 hover:shadow-yellow-500/20",
                    viewMode === "list" && "flex flex-row items-center"
                  )}
                  style={{
                    animationDelay: `${index * 50}ms`,
                    animation: "fadeInUp 0.5s ease-out forwards"
                  }}
                >
                  {/* Status Indicator Line */}
                  <div className={cn(
                    "absolute top-0 left-0 right-0 h-1",
                    status === "active" && "bg-gradient-to-r from-green-400 to-emerald-500",
                    status === "paused" && "bg-gradient-to-r from-yellow-400 to-orange-500",
                    status === "draft" && "bg-gradient-to-r from-gray-300 to-gray-400"
                  )} />

                  <CardHeader className={cn("pb-3", viewMode === "list" && "flex-shrink-0 w-auto")}>
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex items-center gap-4">
                        <div className={cn(
                          "relative flex h-14 w-14 items-center justify-center rounded-2xl transition-transform group-hover:scale-105",
                          status === "active" && "bg-gradient-to-br from-green-100 to-emerald-50 dark:from-green-900/30 dark:to-emerald-900/20",
                          status === "paused" && "bg-gradient-to-br from-yellow-100 to-orange-50 dark:from-yellow-900/30 dark:to-orange-900/20",
                          status === "draft" && "bg-gradient-to-br from-gray-100 to-slate-50 dark:from-gray-800/50 dark:to-slate-800/30"
                        )}>
                          {status === "active" && (
                            <span className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-background animate-pulse" />
                          )}
                          <Bot className={cn(
                            "h-7 w-7",
                            status === "active" && "text-green-600 dark:text-green-400",
                            status === "paused" && "text-yellow-600 dark:text-yellow-400",
                            status === "draft" && "text-gray-500"
                          )} />
                        </div>
                        <div>
                          <CardTitle className="text-lg font-semibold group-hover:text-primary transition-colors">
                            {agent.name}
                          </CardTitle>
                          <Badge className={cn(
                            "mt-1 font-medium",
                            status === "active" && "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
                            status === "paused" && "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
                            status === "draft" && "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
                          )}>
                            {status === "active" && <Activity className="w-3 h-3 mr-1" />}
                            {status}
                          </Badge>
                        </div>
                      </div>
                      <Button variant="ghost" size="icon-sm" className="rounded-xl opacity-0 group-hover:opacity-100 transition-opacity">
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardHeader>

                  <CardContent className={cn(viewMode === "list" && "flex-1 flex items-center gap-6 py-3")}>
                    {viewMode === "grid" && (
                      <>
                        <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                          {agent.description || "No description provided"}
                        </p>

                        {/* Config Pills */}
                        <div className="flex flex-wrap gap-2 mb-4">
                          <span className="inline-flex items-center rounded-full bg-primary/10 text-primary px-3 py-1 text-xs font-medium">
                            <Zap className="w-3 h-3 mr-1" />
                            {agent.llm_config?.model || "GPT-4o"}
                          </span>
                          <span className="inline-flex items-center rounded-full bg-accent/10 text-accent px-3 py-1 text-xs font-medium">
                            {agent.voice_config?.voice_name || "Default Voice"}
                          </span>
                        </div>

                        {/* Stats */}
                        {status !== "draft" && (
                          <div className="grid grid-cols-3 gap-3 rounded-2xl bg-gradient-to-br from-muted/50 to-muted/30 p-4">
                            <div className="text-center">
                              <div className="flex items-center justify-center gap-1 text-muted-foreground mb-1">
                                <Phone className="h-4 w-4" />
                              </div>
                              <p className="text-xl font-bold">{formatNumber(agent.total_calls || 0)}</p>
                              <p className="text-xs text-muted-foreground">calls</p>
                            </div>
                            <div className="text-center border-x border-border/50">
                              <div className="flex items-center justify-center gap-1 text-muted-foreground mb-1">
                                <TrendingUp className="h-4 w-4" />
                              </div>
                              <p className="text-xl font-bold text-green-600">
                                {((agent.success_rate || 0) * 100).toFixed(0)}%
                              </p>
                              <p className="text-xs text-muted-foreground">success</p>
                            </div>
                            <div className="text-center">
                              <div className="flex items-center justify-center gap-1 text-muted-foreground mb-1">
                                <Clock className="h-4 w-4" />
                              </div>
                              <p className="text-xl font-bold">{Math.round((agent.avg_duration || 0) / 60)}m</p>
                              <p className="text-xs text-muted-foreground">avg</p>
                            </div>
                          </div>
                        )}

                        {/* Actions */}
                        <div className="flex gap-2 mt-4">
                          <Button variant="outline" size="sm" className="flex-1 rounded-xl h-10" asChild>
                            <Link href={`/agents/${agent.id}`}>
                              <Settings2 className="mr-2 h-4 w-4" />
                              Configure
                            </Link>
                          </Button>
                          {status === "active" ? (
                            <Button variant="outline" size="sm" className="flex-1 rounded-xl h-10 border-yellow-500/50 text-yellow-600 hover:bg-yellow-50 dark:hover:bg-yellow-900/20">
                              <Pause className="mr-2 h-4 w-4" />
                              Pause
                            </Button>
                          ) : (
                            <Button variant="outline" size="sm" className="flex-1 rounded-xl h-10 border-green-500/50 text-green-600 hover:bg-green-50 dark:hover:bg-green-900/20">
                              <Play className="mr-2 h-4 w-4" />
                              Activate
                            </Button>
                          )}
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            className="rounded-xl"
                            onClick={() => duplicateMutation.mutate(agent.id)}
                            disabled={duplicateMutation.isPending}
                          >
                            {duplicateMutation.isPending ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Copy className="h-4 w-4" />
                            )}
                          </Button>
                        </div>

                        {/* Footer */}
                        <p className="text-xs text-muted-foreground mt-4 pt-3 border-t border-border/50">
                          Updated {formatRelativeTime(agent.updated_at || agent.created_at)}
                        </p>
                      </>
                    )}

                    {viewMode === "list" && (
                      <>
                        <p className="flex-1 text-sm text-muted-foreground line-clamp-1">
                          {agent.description || "No description provided"}
                        </p>
                        <div className="flex items-center gap-6 text-sm">
                          <div className="text-center">
                            <span className="font-semibold">{formatNumber(agent.total_calls || 0)}</span>
                            <span className="text-muted-foreground ml-1">calls</span>
                          </div>
                          <div className="text-center">
                            <span className="font-semibold text-green-600">{((agent.success_rate || 0) * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <Button variant="outline" size="sm" className="rounded-xl" asChild>
                            <Link href={`/agents/${agent.id}`}>
                              <Settings2 className="mr-2 h-4 w-4" />
                              Configure
                            </Link>
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            className="rounded-xl"
                            onClick={() => duplicateMutation.mutate(agent.id)}
                          >
                            <Copy className="h-4 w-4" />
                          </Button>
                        </div>
                      </>
                    )}
                  </CardContent>
                </Card>
              );
            })}

            {/* Create New Card */}
            <Link href="/agents/new">
              <Card className={cn(
                "flex items-center justify-center border-2 border-dashed border-border/50 hover:border-primary/50 hover:bg-primary/5 cursor-pointer transition-all duration-300 group",
                viewMode === "grid" ? "min-h-[400px]" : "min-h-[100px]"
              )}>
                <div className={cn("text-center", viewMode === "list" && "flex items-center gap-4")}>
                  <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 to-accent/10 group-hover:from-primary/30 group-hover:to-accent/20 transition-colors">
                    <Plus className="h-8 w-8 text-primary" />
                  </div>
                  <div className={viewMode === "list" ? "text-left" : ""}>
                    <p className="font-semibold text-lg mt-3 group-hover:text-primary transition-colors">Create New Agent</p>
                    <p className="text-sm text-muted-foreground">
                      Build a custom AI voice agent
                    </p>
                  </div>
                </div>
              </Card>
            </Link>
          </div>

          {/* Empty State */}
          {filteredAgents.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center rounded-3xl border-2 border-dashed border-border/50 p-16 bg-gradient-to-br from-muted/30 to-transparent">
              <div className="relative mb-6">
                <div className="absolute -inset-4 bg-primary/10 rounded-full blur-2xl" />
                <div className="relative w-20 h-20 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/10 flex items-center justify-center">
                  <Bot className="h-10 w-10 text-primary" />
                </div>
              </div>
              <h3 className="text-xl font-semibold mb-2">No agents found</h3>
              <p className="text-muted-foreground mb-6 text-center max-w-md">
                {searchQuery
                  ? "Try adjusting your search or filters to find what you're looking for"
                  : "Get started by creating your first AI voice agent"}
              </p>
              <Button asChild size="lg" className="rounded-xl">
                <Link href="/agents/new">
                  <Plus className="mr-2 h-5 w-5" />
                  Create Your First Agent
                </Link>
              </Button>
            </div>
          )}
        </>
      )}

      {/* Add animation keyframes */}
      <style jsx global>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}

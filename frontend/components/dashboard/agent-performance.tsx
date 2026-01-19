"use client";

import React from "react";
import Link from "next/link";
import { Bot, TrendingUp, TrendingDown, Phone, Clock, Star, MoreHorizontal } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Progress } from "@/components/ui/progress";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import { cn, formatNumber, formatDuration, getInitials } from "@/lib/utils";

interface AgentPerformance {
  id: string;
  name: string;
  is_active: boolean;
  total_calls: number;
  successful_calls: number;
  average_duration: number;
  success_rate: number;
  trend: number;
  avatar_color?: string;
}

interface AgentPerformanceCardProps {
  agents: AgentPerformance[];
  isLoading?: boolean;
  className?: string;
}

const agentColors = [
  "bg-blue-500",
  "bg-green-500",
  "bg-purple-500",
  "bg-orange-500",
  "bg-pink-500",
  "bg-cyan-500",
];

export function AgentPerformanceCard({
  agents,
  isLoading,
  className,
}: AgentPerformanceCardProps) {
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-40" />
            <Skeleton className="h-8 w-20" />
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="flex items-center gap-4 p-3 rounded-lg border">
              <Skeleton className="h-10 w-10 rounded-full" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-3 w-24" />
              </div>
              <Skeleton className="h-6 w-16" />
            </div>
          ))}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-5 w-5" />
            Agent Performance
          </CardTitle>
          <Button variant="ghost" size="sm" asChild>
            <Link href="/agents">View All</Link>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {agents.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <div className="rounded-full bg-muted p-4 mb-4">
              <Bot className="h-8 w-8 text-muted-foreground" />
            </div>
            <p className="font-medium">No agents yet</p>
            <p className="text-sm text-muted-foreground mt-1">
              Create your first agent to see performance metrics
            </p>
            <Button className="mt-4" asChild>
              <Link href="/agents">Create Agent</Link>
            </Button>
          </div>
        ) : (
          <div className="space-y-3">
            {agents.map((agent, index) => (
              <AgentRow
                key={agent.id}
                agent={agent}
                rank={index + 1}
                color={agentColors[index % agentColors.length]}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface AgentRowProps {
  agent: AgentPerformance;
  rank: number;
  color: string;
}

function AgentRow({ agent, rank, color }: AgentRowProps) {
  return (
    <div className="group flex items-center gap-3 p-3 rounded-lg border hover:border-primary/30 hover:bg-muted/50 transition-all">
      <div className="flex items-center gap-3 flex-1 min-w-0">
        <div className="relative">
          <Avatar className="h-10 w-10">
            <AvatarFallback className={cn(color, "text-white")}>
              {getInitials(agent.name)}
            </AvatarFallback>
          </Avatar>
          {agent.is_active && (
            <span className="absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full bg-green-500 border-2 border-background" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium truncate">{agent.name}</span>
            {rank <= 3 && (
              <Star className={cn(
                "h-3.5 w-3.5",
                rank === 1 ? "text-yellow-500 fill-yellow-500" :
                rank === 2 ? "text-gray-400 fill-gray-400" :
                "text-amber-600 fill-amber-600"
              )} />
            )}
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
            <span className="flex items-center gap-1">
              <Phone className="h-3 w-3" />
              {formatNumber(agent.total_calls)} calls
            </span>
            <span className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {formatDuration(agent.average_duration)} avg
            </span>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <div className="text-right hidden sm:block">
          <div className="text-sm font-medium">{agent.success_rate.toFixed(1)}%</div>
          <div className={cn(
            "flex items-center justify-end gap-1 text-xs",
            agent.trend >= 0 ? "text-green-600" : "text-red-600"
          )}>
            {agent.trend >= 0 ? (
              <TrendingUp className="h-3 w-3" />
            ) : (
              <TrendingDown className="h-3 w-3" />
            )}
            {Math.abs(agent.trend)}%
          </div>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              className="opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem asChild>
              <Link href={`/agents/${agent.id}`}>View Details</Link>
            </DropdownMenuItem>
            <DropdownMenuItem asChild>
              <Link href={`/agents/${agent.id}/edit`}>Edit Agent</Link>
            </DropdownMenuItem>
            <DropdownMenuItem asChild>
              <Link href={`/calls?agent=${agent.id}`}>View Calls</Link>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}

interface AgentLeaderboardProps {
  agents: AgentPerformance[];
  isLoading?: boolean;
  className?: string;
}

export function AgentLeaderboard({ agents, isLoading, className }: AgentLeaderboardProps) {
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <Skeleton className="h-6 w-32" />
        </CardHeader>
        <CardContent className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="flex items-center gap-4">
              <Skeleton className="h-8 w-8 rounded-full" />
              <div className="flex-1">
                <Skeleton className="h-4 w-full mb-2" />
                <Skeleton className="h-2 w-full" />
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    );
  }

  const topAgents = agents.slice(0, 5);
  const maxCalls = Math.max(...topAgents.map((a) => a.total_calls), 1);

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-base">Top Performers</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {topAgents.map((agent, index) => (
          <div key={agent.id} className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <span className="font-medium text-muted-foreground w-4">
                  #{index + 1}
                </span>
                <span className="font-medium truncate">{agent.name}</span>
              </div>
              <span className="text-muted-foreground">
                {formatNumber(agent.total_calls)} calls
              </span>
            </div>
            <Progress
              value={(agent.total_calls / maxCalls) * 100}
              className="h-1.5"
            />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

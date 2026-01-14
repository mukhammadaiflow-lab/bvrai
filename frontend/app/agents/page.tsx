"use client";

import React, { useState } from "react";
import Link from "next/link";
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
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn, formatRelativeTime, formatNumber, getStatusColor } from "@/lib/utils";

// Mock agents data
const mockAgents = [
  {
    id: "1",
    name: "Sales Agent",
    description: "Handles inbound sales inquiries and qualifies leads",
    status: "active",
    llm: "GPT-4o",
    voice: "Rachel (ElevenLabs)",
    totalCalls: 1247,
    successRate: 0.94,
    avgDuration: 185,
    createdAt: "2024-01-01T00:00:00Z",
    updatedAt: "2024-01-14T10:00:00Z",
  },
  {
    id: "2",
    name: "Support Agent",
    description: "Customer support and technical assistance",
    status: "active",
    llm: "GPT-4o",
    voice: "Josh (ElevenLabs)",
    totalCalls: 892,
    successRate: 0.91,
    avgDuration: 240,
    createdAt: "2024-01-05T00:00:00Z",
    updatedAt: "2024-01-14T09:30:00Z",
  },
  {
    id: "3",
    name: "Booking Agent",
    description: "Appointment scheduling and calendar management",
    status: "paused",
    llm: "GPT-3.5-turbo",
    voice: "Bella (ElevenLabs)",
    totalCalls: 456,
    successRate: 0.88,
    avgDuration: 120,
    createdAt: "2024-01-08T00:00:00Z",
    updatedAt: "2024-01-13T16:00:00Z",
  },
  {
    id: "4",
    name: "Survey Agent",
    description: "Conducts customer satisfaction surveys",
    status: "draft",
    llm: "GPT-4o",
    voice: "Adam (ElevenLabs)",
    totalCalls: 0,
    successRate: 0,
    avgDuration: 0,
    createdAt: "2024-01-12T00:00:00Z",
    updatedAt: "2024-01-12T00:00:00Z",
  },
];

export default function AgentsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string | null>(null);

  const filteredAgents = mockAgents.filter((agent) => {
    const matchesSearch =
      agent.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      agent.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = !statusFilter || agent.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const statusCounts = {
    all: mockAgents.length,
    active: mockAgents.filter((a) => a.status === "active").length,
    paused: mockAgents.filter((a) => a.status === "paused").length,
    draft: mockAgents.filter((a) => a.status === "draft").length,
  };

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
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          Create Agent
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

      {/* Agents Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filteredAgents.map((agent) => (
          <Card key={agent.id} className="group relative overflow-hidden">
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className={cn(
                    "flex h-12 w-12 items-center justify-center rounded-lg",
                    agent.status === "active" ? "bg-green-100" :
                    agent.status === "paused" ? "bg-yellow-100" :
                    "bg-gray-100"
                  )}>
                    <Bot className={cn(
                      "h-6 w-6",
                      agent.status === "active" ? "text-green-600" :
                      agent.status === "paused" ? "text-yellow-600" :
                      "text-gray-600"
                    )} />
                  </div>
                  <div>
                    <CardTitle className="text-lg">{agent.name}</CardTitle>
                    <Badge className={getStatusColor(agent.status)}>
                      {agent.status}
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
                {agent.description}
              </p>

              {/* Config Pills */}
              <div className="flex flex-wrap gap-2 mb-4">
                <span className="inline-flex items-center rounded-full bg-secondary px-2.5 py-0.5 text-xs">
                  {agent.llm}
                </span>
                <span className="inline-flex items-center rounded-full bg-secondary px-2.5 py-0.5 text-xs">
                  {agent.voice}
                </span>
              </div>

              {/* Stats */}
              {agent.status !== "draft" && (
                <div className="grid grid-cols-3 gap-2 rounded-lg bg-secondary/50 p-3">
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-1 text-muted-foreground">
                      <Phone className="h-3 w-3" />
                    </div>
                    <p className="text-lg font-bold">{formatNumber(agent.totalCalls)}</p>
                    <p className="text-xs text-muted-foreground">calls</p>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-1 text-muted-foreground">
                      <TrendingUp className="h-3 w-3" />
                    </div>
                    <p className="text-lg font-bold text-green-600">
                      {(agent.successRate * 100).toFixed(0)}%
                    </p>
                    <p className="text-xs text-muted-foreground">success</p>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-1 text-muted-foreground">
                      <Clock className="h-3 w-3" />
                    </div>
                    <p className="text-lg font-bold">{Math.round(agent.avgDuration / 60)}m</p>
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
                {agent.status === "active" ? (
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
                <Button variant="ghost" size="icon-sm">
                  <Copy className="h-3 w-3" />
                </Button>
              </div>

              {/* Footer */}
              <p className="text-xs text-muted-foreground mt-3">
                Updated {formatRelativeTime(agent.updatedAt)}
              </p>
            </CardContent>
          </Card>
        ))}

        {/* Create New Card */}
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
      </div>

      {/* Empty State */}
      {filteredAgents.length === 0 && (
        <div className="flex flex-col items-center justify-center rounded-lg border border-dashed p-12">
          <Bot className="h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">No agents found</h3>
          <p className="text-muted-foreground mb-4">
            {searchQuery
              ? "Try adjusting your search or filters"
              : "Get started by creating your first agent"}
          </p>
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            Create Agent
          </Button>
        </div>
      )}
    </div>
  );
}

"use client";

import React, { useState } from "react";
import {
  Webhook,
  Plus,
  Search,
  MoreHorizontal,
  CheckCircle,
  XCircle,
  Clock,
  ExternalLink,
  Copy,
  Trash2,
  Edit,
  Play,
  RefreshCw,
  AlertTriangle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn, formatRelativeTime } from "@/lib/utils";

// Mock webhooks data
const mockWebhooks = [
  {
    id: "1",
    name: "CRM Integration",
    url: "https://api.example.com/webhooks/voice-ai",
    events: ["call.started", "call.ended", "transfer.completed"],
    status: "active",
    lastTriggered: "2024-01-14T10:30:00Z",
    successRate: 98.5,
    totalDeliveries: 1247,
    failedDeliveries: 19,
  },
  {
    id: "2",
    name: "Analytics Pipeline",
    url: "https://analytics.example.com/ingest",
    events: ["call.ended", "conversation.message"],
    status: "active",
    lastTriggered: "2024-01-14T10:25:00Z",
    successRate: 100,
    totalDeliveries: 892,
    failedDeliveries: 0,
  },
  {
    id: "3",
    name: "Slack Notifications",
    url: "https://hooks.slack.com/services/xxx/yyy/zzz",
    events: ["call.failed", "transfer.initiated"],
    status: "active",
    lastTriggered: "2024-01-14T09:15:00Z",
    successRate: 95.2,
    totalDeliveries: 312,
    failedDeliveries: 15,
  },
  {
    id: "4",
    name: "Legacy System",
    url: "https://old-api.example.com/callback",
    events: ["call.ended"],
    status: "failing",
    lastTriggered: "2024-01-14T08:00:00Z",
    successRate: 45.0,
    totalDeliveries: 156,
    failedDeliveries: 86,
  },
];

const availableEvents = [
  { id: "call.started", label: "Call Started", description: "Triggered when a call begins" },
  { id: "call.ended", label: "Call Ended", description: "Triggered when a call completes" },
  { id: "call.failed", label: "Call Failed", description: "Triggered when a call fails" },
  { id: "conversation.message", label: "Conversation Message", description: "Triggered for each message" },
  { id: "conversation.ended", label: "Conversation Ended", description: "Triggered when conversation ends" },
  { id: "transfer.initiated", label: "Transfer Initiated", description: "Triggered when transfer starts" },
  { id: "transfer.completed", label: "Transfer Completed", description: "Triggered when transfer completes" },
  { id: "agent.updated", label: "Agent Updated", description: "Triggered when agent config changes" },
];

export default function WebhooksPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [showCreateModal, setShowCreateModal] = useState(false);

  const filteredWebhooks = mockWebhooks.filter((webhook) =>
    webhook.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    webhook.url.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Webhooks</h1>
          <p className="text-muted-foreground">
            Receive real-time notifications for voice AI events
          </p>
        </div>
        <Button onClick={() => setShowCreateModal(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Add Webhook
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                <Webhook className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{mockWebhooks.length}</p>
                <p className="text-sm text-muted-foreground">Total Webhooks</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-100">
                <CheckCircle className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">2,607</p>
                <p className="text-sm text-muted-foreground">Deliveries (24h)</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-100">
                <Clock className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">145ms</p>
                <p className="text-sm text-muted-foreground">Avg Response Time</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-100">
                <XCircle className="h-5 w-5 text-red-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">120</p>
                <p className="text-sm text-muted-foreground">Failed (24h)</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <div className="flex gap-4">
        <Input
          placeholder="Search webhooks..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          leftIcon={<Search className="h-4 w-4" />}
          className="max-w-md"
        />
      </div>

      {/* Webhooks List */}
      <div className="space-y-4">
        {filteredWebhooks.map((webhook) => (
          <Card key={webhook.id}>
            <CardContent className="p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4">
                  <div className={cn(
                    "flex h-12 w-12 items-center justify-center rounded-lg",
                    webhook.status === "active" ? "bg-green-100" : "bg-red-100"
                  )}>
                    <Webhook className={cn(
                      "h-6 w-6",
                      webhook.status === "active" ? "text-green-600" : "text-red-600"
                    )} />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-semibold">{webhook.name}</h3>
                      <Badge
                        className={cn(
                          webhook.status === "active"
                            ? "bg-green-100 text-green-800"
                            : "bg-red-100 text-red-800"
                        )}
                      >
                        {webhook.status === "active" ? "Active" : "Failing"}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground mb-3">
                      <code className="bg-muted px-2 py-0.5 rounded text-xs">{webhook.url}</code>
                      <Button variant="ghost" size="icon-sm">
                        <Copy className="h-3 w-3" />
                      </Button>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {webhook.events.map((event) => (
                        <Badge key={event} variant="outline" className="text-xs">
                          {event}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="sm">
                    <Play className="mr-1 h-4 w-4" />
                    Test
                  </Button>
                  <Button variant="ghost" size="sm">
                    <Edit className="mr-1 h-4 w-4" />
                    Edit
                  </Button>
                  <Button variant="ghost" size="icon-sm">
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <div className="mt-4 pt-4 border-t">
                <div className="grid grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Last Triggered</p>
                    <p className="font-medium">{formatRelativeTime(webhook.lastTriggered)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Success Rate</p>
                    <p className={cn(
                      "font-medium",
                      webhook.successRate >= 95 ? "text-green-600" :
                      webhook.successRate >= 80 ? "text-yellow-600" : "text-red-600"
                    )}>
                      {webhook.successRate.toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Total Deliveries</p>
                    <p className="font-medium">{webhook.totalDeliveries.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Failed</p>
                    <p className={cn(
                      "font-medium",
                      webhook.failedDeliveries > 50 ? "text-red-600" : ""
                    )}>
                      {webhook.failedDeliveries}
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Available Events Reference */}
      <Card>
        <CardHeader>
          <CardTitle>Available Events</CardTitle>
          <CardDescription>
            Events you can subscribe to with your webhooks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            {availableEvents.map((event) => (
              <div key={event.id} className="rounded-lg border p-3">
                <code className="text-sm font-medium">{event.id}</code>
                <p className="text-xs text-muted-foreground mt-1">{event.description}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

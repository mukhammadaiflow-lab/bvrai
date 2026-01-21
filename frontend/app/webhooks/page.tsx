"use client";

import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { DashboardLayout } from "@/components/layouts/dashboard-layout";
import { webhooks as webhooksApi } from "@/lib/api";
import { toast } from "sonner";
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
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  Label,
  Switch,
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
  Checkbox,
  Separator,
  ScrollArea,
  Skeleton,
  Textarea,
} from "@/components/ui";
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
  Eye,
  EyeOff,
  Shield,
  Code,
  Send,
  ChevronDown,
  ChevronUp,
  Zap,
  Activity,
  Settings,
  AlertCircle,
  Check,
  X,
  RotateCw,
  ArrowRight,
  FileJson,
  History,
  Loader2,
  Terminal,
  Globe,
  Lock,
} from "lucide-react";
import { cn } from "@/lib/utils";

// Types
interface WebhookEvent {
  id: string;
  label: string;
  description: string;
  category: string;
}

interface Webhook {
  id: string;
  name: string;
  url: string;
  events: string[];
  status: "active" | "failing" | "paused" | "disabled";
  secret?: string;
  headers?: Record<string, string>;
  retryPolicy: {
    maxRetries: number;
    retryDelay: number;
  };
  lastTriggered: string;
  successRate: number;
  totalDeliveries: number;
  failedDeliveries: number;
  created_at: string;
}

interface DeliveryLog {
  id: string;
  webhook_id: string;
  event_type: string;
  status: "success" | "failed" | "pending" | "retrying";
  status_code?: number;
  response_time?: number;
  request_body: string;
  response_body?: string;
  error_message?: string;
  timestamp: string;
  retries: number;
}

// Mock data
const availableEvents: WebhookEvent[] = [
  { id: "call.started", label: "Call Started", description: "Triggered when a call begins", category: "Calls" },
  { id: "call.ended", label: "Call Ended", description: "Triggered when a call completes", category: "Calls" },
  { id: "call.failed", label: "Call Failed", description: "Triggered when a call fails", category: "Calls" },
  { id: "call.transferred", label: "Call Transferred", description: "Triggered when a call is transferred", category: "Calls" },
  { id: "conversation.message", label: "Message Sent", description: "Triggered for each message in a conversation", category: "Conversations" },
  { id: "conversation.ended", label: "Conversation Ended", description: "Triggered when a conversation ends", category: "Conversations" },
  { id: "conversation.sentiment", label: "Sentiment Changed", description: "Triggered when sentiment changes significantly", category: "Conversations" },
  { id: "agent.created", label: "Agent Created", description: "Triggered when a new agent is created", category: "Agents" },
  { id: "agent.updated", label: "Agent Updated", description: "Triggered when agent configuration changes", category: "Agents" },
  { id: "agent.deleted", label: "Agent Deleted", description: "Triggered when an agent is deleted", category: "Agents" },
  { id: "transcript.ready", label: "Transcript Ready", description: "Triggered when call transcript is available", category: "Transcripts" },
  { id: "recording.ready", label: "Recording Ready", description: "Triggered when call recording is processed", category: "Recordings" },
];

// Transform API webhook to component format
const transformWebhook = (apiWebhook: any): Webhook => ({
  id: apiWebhook.id,
  name: apiWebhook.name,
  url: apiWebhook.url,
  events: apiWebhook.events || [],
  status: apiWebhook.is_active ? (apiWebhook.failure_count > 10 ? "failing" : "active") : "disabled",
  secret: apiWebhook.secret,
  headers: apiWebhook.headers,
  retryPolicy: {
    maxRetries: apiWebhook.max_retries || 3,
    retryDelay: apiWebhook.retry_delay_ms || 5000,
  },
  lastTriggered: apiWebhook.last_triggered_at || apiWebhook.updated_at,
  successRate: apiWebhook.success_rate || 100,
  totalDeliveries: apiWebhook.total_deliveries || 0,
  failedDeliveries: apiWebhook.failed_deliveries || 0,
  created_at: apiWebhook.created_at,
});

// Utility functions
const formatRelativeTime = (date: string): string => {
  const now = new Date();
  const past = new Date(date);
  const diffMs = now.getTime() - past.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return past.toLocaleDateString();
};

// Components
function WebhookStatusBadge({ status }: { status: Webhook["status"] }) {
  const config = {
    active: { color: "bg-green-100 text-green-700 border-green-200", icon: CheckCircle, label: "Active" },
    failing: { color: "bg-red-100 text-red-700 border-red-200", icon: XCircle, label: "Failing" },
    paused: { color: "bg-yellow-100 text-yellow-700 border-yellow-200", icon: Clock, label: "Paused" },
    disabled: { color: "bg-gray-100 text-gray-700 border-gray-200", icon: XCircle, label: "Disabled" },
  };

  const { color, icon: Icon, label } = config[status];

  return (
    <Badge className={cn("gap-1", color)}>
      <Icon className="h-3 w-3" />
      {label}
    </Badge>
  );
}

function DeliveryStatusBadge({ status }: { status: DeliveryLog["status"] }) {
  const config = {
    success: { color: "bg-green-100 text-green-700", icon: Check, label: "Success" },
    failed: { color: "bg-red-100 text-red-700", icon: X, label: "Failed" },
    pending: { color: "bg-yellow-100 text-yellow-700", icon: Clock, label: "Pending" },
    retrying: { color: "bg-blue-100 text-blue-700", icon: RotateCw, label: "Retrying" },
  };

  const { color, icon: Icon, label } = config[status];

  return (
    <Badge className={cn("gap-1", color)}>
      <Icon className="h-3 w-3" />
      {label}
    </Badge>
  );
}

function CreateWebhookDialog({
  open,
  onOpenChange,
  webhook,
  onSave,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  webhook?: Webhook | null;
  onSave: (data: Partial<Webhook>) => void;
}) {
  const [name, setName] = useState(webhook?.name || "");
  const [url, setUrl] = useState(webhook?.url || "");
  const [selectedEvents, setSelectedEvents] = useState<string[]>(webhook?.events || []);
  const [secret, setSecret] = useState(webhook?.secret || "");
  const [showSecret, setShowSecret] = useState(false);
  const [maxRetries, setMaxRetries] = useState(webhook?.retryPolicy.maxRetries || 3);
  const [retryDelay, setRetryDelay] = useState(webhook?.retryPolicy.retryDelay || 5000);

  const isEdit = !!webhook;

  const handleToggleEvent = (eventId: string) => {
    setSelectedEvents((prev) =>
      prev.includes(eventId)
        ? prev.filter((e) => e !== eventId)
        : [...prev, eventId]
    );
  };

  const handleSelectAllInCategory = (category: string) => {
    const categoryEvents = availableEvents.filter((e) => e.category === category).map((e) => e.id);
    const allSelected = categoryEvents.every((e) => selectedEvents.includes(e));

    if (allSelected) {
      setSelectedEvents((prev) => prev.filter((e) => !categoryEvents.includes(e)));
    } else {
      setSelectedEvents((prev) => [...new Set([...prev, ...categoryEvents])]);
    }
  };

  const handleSave = () => {
    onSave({
      name,
      url,
      events: selectedEvents,
      secret: secret || undefined,
      retryPolicy: { maxRetries, retryDelay },
    });
    onOpenChange(false);
  };

  const categories = Array.from(new Set(availableEvents.map((e) => e.category)));

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>{isEdit ? "Edit Webhook" : "Create New Webhook"}</DialogTitle>
          <DialogDescription>
            Configure your webhook endpoint to receive real-time event notifications.
          </DialogDescription>
        </DialogHeader>

        <ScrollArea className="flex-1 pr-4">
          <div className="space-y-6 py-4">
            {/* Basic Info */}
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  placeholder="My Webhook"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="url">Endpoint URL</Label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Globe className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="url"
                      placeholder="https://api.example.com/webhooks"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      className="pl-9"
                    />
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">
                  Must be a publicly accessible HTTPS endpoint
                </p>
              </div>
            </div>

            <Separator />

            {/* Events */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>Events to subscribe</Label>
                <span className="text-sm text-muted-foreground">
                  {selectedEvents.length} selected
                </span>
              </div>

              <div className="space-y-4">
                {categories.map((category) => {
                  const categoryEvents = availableEvents.filter((e) => e.category === category);
                  const selectedCount = categoryEvents.filter((e) =>
                    selectedEvents.includes(e.id)
                  ).length;

                  return (
                    <div key={category} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">{category}</span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleSelectAllInCategory(category)}
                          className="text-xs h-6"
                        >
                          {selectedCount === categoryEvents.length ? "Deselect all" : "Select all"}
                        </Button>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        {categoryEvents.map((event) => (
                          <div
                            key={event.id}
                            className={cn(
                              "flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors",
                              selectedEvents.includes(event.id)
                                ? "border-primary bg-primary/5"
                                : "hover:bg-muted/50"
                            )}
                            onClick={() => handleToggleEvent(event.id)}
                          >
                            <Checkbox
                              checked={selectedEvents.includes(event.id)}
                              onCheckedChange={() => handleToggleEvent(event.id)}
                            />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium">{event.label}</p>
                              <p className="text-xs text-muted-foreground truncate">
                                {event.description}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <Separator />

            {/* Security */}
            <div className="space-y-4">
              <Label>Security</Label>
              <div className="space-y-2">
                <Label htmlFor="secret" className="text-sm font-normal text-muted-foreground">
                  Signing Secret (optional)
                </Label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="secret"
                      type={showSecret ? "text" : "password"}
                      placeholder="whsec_..."
                      value={secret}
                      onChange={(e) => setSecret(e.target.value)}
                      className="pl-9 pr-10 font-mono text-sm"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="absolute right-1 top-1/2 -translate-y-1/2 h-7 w-7"
                      onClick={() => setShowSecret(!showSecret)}
                    >
                      {showSecret ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                  <Button
                    variant="outline"
                    onClick={() =>
                      setSecret(`whsec_${Math.random().toString(36).substring(2, 15)}`)
                    }
                  >
                    Generate
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  We'll include a signature header for request verification
                </p>
              </div>
            </div>

            <Separator />

            {/* Retry Policy */}
            <div className="space-y-4">
              <Label>Retry Policy</Label>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="maxRetries" className="text-sm font-normal text-muted-foreground">
                    Max Retries
                  </Label>
                  <Select
                    value={maxRetries.toString()}
                    onValueChange={(v) => setMaxRetries(parseInt(v))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {[0, 1, 2, 3, 5, 10].map((n) => (
                        <SelectItem key={n} value={n.toString()}>
                          {n} {n === 1 ? "retry" : "retries"}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="retryDelay" className="text-sm font-normal text-muted-foreground">
                    Retry Delay
                  </Label>
                  <Select
                    value={retryDelay.toString()}
                    onValueChange={(v) => setRetryDelay(parseInt(v))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1000">1 second</SelectItem>
                      <SelectItem value="3000">3 seconds</SelectItem>
                      <SelectItem value="5000">5 seconds</SelectItem>
                      <SelectItem value="10000">10 seconds</SelectItem>
                      <SelectItem value="30000">30 seconds</SelectItem>
                      <SelectItem value="60000">1 minute</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </div>
        </ScrollArea>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={!name || !url || selectedEvents.length === 0}
          >
            {isEdit ? "Save Changes" : "Create Webhook"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function TestWebhookDialog({
  open,
  onOpenChange,
  webhook,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  webhook: Webhook;
}) {
  const [selectedEvent, setSelectedEvent] = useState(webhook.events[0] || "");
  const [isSending, setIsSending] = useState(false);
  const [testResult, setTestResult] = useState<{
    success: boolean;
    statusCode?: number;
    responseTime?: number;
    response?: string;
    error?: string;
  } | null>(null);

  const samplePayload = {
    event: selectedEvent,
    timestamp: new Date().toISOString(),
    data: {
      call_id: "test-call-123",
      agent_id: "agent-1",
      caller_number: "+15551234567",
      duration: 245,
    },
  };

  const handleTest = async () => {
    setIsSending(true);
    setTestResult(null);

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1500));

    // Mock response
    setTestResult({
      success: Math.random() > 0.2,
      statusCode: Math.random() > 0.2 ? 200 : 500,
      responseTime: Math.floor(Math.random() * 300) + 50,
      response: '{"status": "received", "message": "Webhook test successful"}',
      error: Math.random() > 0.2 ? undefined : "Connection timeout",
    });

    setIsSending(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Test Webhook</DialogTitle>
          <DialogDescription>
            Send a test event to {webhook.name} ({webhook.url})
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Event Selection */}
          <div className="space-y-2">
            <Label>Event Type</Label>
            <Select value={selectedEvent} onValueChange={setSelectedEvent}>
              <SelectTrigger>
                <SelectValue placeholder="Select an event" />
              </SelectTrigger>
              <SelectContent>
                {webhook.events.map((event) => (
                  <SelectItem key={event} value={event}>
                    {event}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Payload Preview */}
          <div className="space-y-2">
            <Label>Request Payload</Label>
            <div className="relative">
              <pre className="p-4 rounded-lg bg-muted text-sm font-mono overflow-auto max-h-48">
                {JSON.stringify(samplePayload, null, 2)}
              </pre>
              <Button
                variant="ghost"
                size="icon"
                className="absolute top-2 right-2"
                onClick={() => navigator.clipboard.writeText(JSON.stringify(samplePayload, null, 2))}
              >
                <Copy className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Test Result */}
          {testResult && (
            <div
              className={cn(
                "p-4 rounded-lg border",
                testResult.success
                  ? "bg-green-50 border-green-200 dark:bg-green-950/20 dark:border-green-800"
                  : "bg-red-50 border-red-200 dark:bg-red-950/20 dark:border-red-800"
              )}
            >
              <div className="flex items-center gap-2 mb-3">
                {testResult.success ? (
                  <CheckCircle className="h-5 w-5 text-green-600" />
                ) : (
                  <XCircle className="h-5 w-5 text-red-600" />
                )}
                <span className="font-medium">
                  {testResult.success ? "Test Successful" : "Test Failed"}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Status Code</p>
                  <p className="font-medium">{testResult.statusCode || "-"}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Response Time</p>
                  <p className="font-medium">{testResult.responseTime}ms</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Result</p>
                  <p className={cn("font-medium", testResult.success ? "text-green-600" : "text-red-600")}>
                    {testResult.success ? "200 OK" : testResult.error}
                  </p>
                </div>
              </div>

              {testResult.response && (
                <div className="mt-3 pt-3 border-t">
                  <p className="text-sm text-muted-foreground mb-1">Response Body</p>
                  <pre className="text-xs bg-background/50 p-2 rounded">
                    {testResult.response}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
          <Button onClick={handleTest} disabled={isSending || !selectedEvent}>
            {isSending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Sending...
              </>
            ) : (
              <>
                <Send className="mr-2 h-4 w-4" />
                Send Test
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function WebhookCard({
  webhook,
  onEdit,
  onTest,
  onDelete,
  onToggleStatus,
}: {
  webhook: Webhook;
  onEdit: () => void;
  onTest: () => void;
  onDelete: () => void;
  onToggleStatus: () => void;
}) {
  const [showSecret, setShowSecret] = useState(false);
  const [expanded, setExpanded] = useState(false);

  return (
    <Card className={cn(webhook.status === "failing" && "border-red-200 bg-red-50/30 dark:bg-red-950/10")}>
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-4">
            <div
              className={cn(
                "flex h-12 w-12 items-center justify-center rounded-lg",
                webhook.status === "active"
                  ? "bg-green-100 dark:bg-green-900/30"
                  : webhook.status === "failing"
                  ? "bg-red-100 dark:bg-red-900/30"
                  : "bg-gray-100 dark:bg-gray-800"
              )}
            >
              <Webhook
                className={cn(
                  "h-6 w-6",
                  webhook.status === "active"
                    ? "text-green-600"
                    : webhook.status === "failing"
                    ? "text-red-600"
                    : "text-gray-600"
                )}
              />
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold">{webhook.name}</h3>
                <WebhookStatusBadge status={webhook.status} />
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <code className="bg-muted px-2 py-0.5 rounded text-xs font-mono max-w-md truncate">
                  {webhook.url}
                </code>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        onClick={() => navigator.clipboard.writeText(webhook.url)}
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Copy URL</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {webhook.events.slice(0, 3).map((event) => (
                  <Badge key={event} variant="outline" className="text-xs">
                    {event}
                  </Badge>
                ))}
                {webhook.events.length > 3 && (
                  <Badge variant="outline" className="text-xs">
                    +{webhook.events.length - 3} more
                  </Badge>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="sm" onClick={onTest}>
                    <Play className="mr-1 h-4 w-4" />
                    Test
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Send a test event</TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <Button variant="outline" size="sm" onClick={onEdit}>
              <Edit className="mr-1 h-4 w-4" />
              Edit
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        {/* Stats Row */}
        <div className="mt-4 pt-4 border-t grid grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Last Triggered</p>
            <p className="font-medium">{formatRelativeTime(webhook.lastTriggered)}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Success Rate</p>
            <p
              className={cn(
                "font-medium",
                webhook.successRate >= 95
                  ? "text-green-600"
                  : webhook.successRate >= 80
                  ? "text-yellow-600"
                  : "text-red-600"
              )}
            >
              {webhook.successRate.toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Total Deliveries</p>
            <p className="font-medium">{webhook.totalDeliveries.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Failed</p>
            <p
              className={cn(
                "font-medium",
                webhook.failedDeliveries > 50 ? "text-red-600" : ""
              )}
            >
              {webhook.failedDeliveries}
            </p>
          </div>
        </div>

        {/* Expanded Content */}
        {expanded && (
          <div className="mt-4 pt-4 border-t space-y-4">
            {/* Secret */}
            {webhook.secret && (
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-2">
                  <Shield className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Signing Secret:</span>
                  <code className="text-sm font-mono">
                    {showSecret ? webhook.secret : "••••••••••••••••"}
                  </code>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => setShowSecret(!showSecret)}
                  >
                    {showSecret ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => navigator.clipboard.writeText(webhook.secret!)}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}

            {/* Retry Policy */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="p-3 bg-muted/50 rounded-lg">
                <p className="text-muted-foreground mb-1">Retry Policy</p>
                <p className="font-medium">
                  {webhook.retryPolicy.maxRetries} retries, {webhook.retryPolicy.retryDelay / 1000}s delay
                </p>
              </div>
              <div className="p-3 bg-muted/50 rounded-lg">
                <p className="text-muted-foreground mb-1">Created</p>
                <p className="font-medium">
                  {new Date(webhook.created_at).toLocaleDateString()}
                </p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-between pt-2">
              <div className="flex items-center gap-2">
                <Switch
                  checked={webhook.status === "active"}
                  onCheckedChange={onToggleStatus}
                />
                <span className="text-sm">
                  {webhook.status === "active" ? "Active" : "Paused"}
                </span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="text-red-600 hover:text-red-700 hover:bg-red-50"
                onClick={onDelete}
              >
                <Trash2 className="mr-1 h-4 w-4" />
                Delete
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function DeliveryLogRow({ log }: { log: DeliveryLog }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border-b last:border-0">
      <div
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-muted/50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-4">
          <DeliveryStatusBadge status={log.status} />
          <div>
            <code className="text-sm font-medium">{log.event_type}</code>
            <p className="text-xs text-muted-foreground">
              {formatRelativeTime(log.timestamp)}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right text-sm">
            <p className="font-medium">{log.status_code || "-"}</p>
            <p className="text-xs text-muted-foreground">
              {log.response_time ? `${log.response_time}ms` : "-"}
            </p>
          </div>
          {log.retries > 0 && (
            <Badge variant="outline" className="text-xs">
              {log.retries} retries
            </Badge>
          )}
          {expanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </div>
      </div>

      {expanded && (
        <div className="px-4 pb-4 space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">Request Body</p>
              <pre className="text-xs bg-muted p-3 rounded-lg overflow-auto max-h-48">
                {log.request_body}
              </pre>
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">Response</p>
              {log.response_body ? (
                <pre className="text-xs bg-muted p-3 rounded-lg overflow-auto max-h-48">
                  {log.response_body}
                </pre>
              ) : log.error_message ? (
                <div className="text-xs bg-red-50 dark:bg-red-950/30 text-red-600 p-3 rounded-lg">
                  {log.error_message}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground p-3">No response</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function WebhooksPage() {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState("");
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [editingWebhook, setEditingWebhook] = useState<Webhook | null>(null);
  const [testingWebhook, setTestingWebhook] = useState<Webhook | null>(null);
  const [activeTab, setActiveTab] = useState("webhooks");
  const [selectedWebhookForLogs, setSelectedWebhookForLogs] = useState<string>("all");

  // Fetch webhooks from API
  const { data: webhooksData, isLoading } = useQuery({
    queryKey: ["webhooks"],
    queryFn: () => webhooksApi.list(),
  });

  // Fetch delivery logs for selected webhook
  const { data: deliveriesData } = useQuery({
    queryKey: ["webhooks", "deliveries", selectedWebhookForLogs],
    queryFn: async () => {
      if (selectedWebhookForLogs !== "all") {
        return webhooksApi.getDeliveries(selectedWebhookForLogs);
      }
      return { items: [], total: 0, page: 1, page_size: 20, total_pages: 0 };
    },
    enabled: selectedWebhookForLogs !== "all",
  });

  // Create webhook mutation
  const createMutation = useMutation({
    mutationFn: (data: Partial<Webhook>) =>
      webhooksApi.create({
        name: data.name,
        url: data.url,
        events: data.events as any,
        secret: data.secret,
        headers: data.headers,
        max_retries: data.retryPolicy?.maxRetries,
        retry_delay_ms: data.retryPolicy?.retryDelay,
      } as any),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["webhooks"] });
      toast.success("Webhook created successfully");
      setShowCreateDialog(false);
    },
    onError: (error: Error) => {
      toast.error(`Failed to create webhook: ${error.message}`);
    },
  });

  // Update webhook mutation
  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<Webhook> }) =>
      webhooksApi.update(id, {
        name: data.name,
        url: data.url,
        events: data.events as any,
        secret: data.secret,
        headers: data.headers,
        max_retries: data.retryPolicy?.maxRetries,
        retry_delay_ms: data.retryPolicy?.retryDelay,
      } as any),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["webhooks"] });
      toast.success("Webhook updated successfully");
      setEditingWebhook(null);
      setShowCreateDialog(false);
    },
    onError: (error: Error) => {
      toast.error(`Failed to update webhook: ${error.message}`);
    },
  });

  // Delete webhook mutation
  const deleteMutation = useMutation({
    mutationFn: (webhookId: string) => webhooksApi.delete(webhookId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["webhooks"] });
      toast.success("Webhook deleted successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete webhook: ${error.message}`);
    },
  });

  // Transform API data
  const webhooks: Webhook[] = (webhooksData || []).map(transformWebhook);

  // Transform delivery logs
  const deliveryLogs: DeliveryLog[] = (deliveriesData?.items || []).map((d: any) => ({
    id: d.id,
    webhook_id: d.webhook_id,
    event_type: d.event_type,
    status: d.status,
    status_code: d.status_code,
    response_time: d.response_time_ms,
    request_body: d.request_body || "{}",
    response_body: d.response_body,
    error_message: d.error_message,
    timestamp: d.created_at,
    retries: d.retry_count || 0,
  }));

  const filteredWebhooks = webhooks.filter(
    (webhook) =>
      webhook.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      webhook.url.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredLogs =
    selectedWebhookForLogs === "all"
      ? deliveryLogs
      : deliveryLogs.filter((log) => log.webhook_id === selectedWebhookForLogs);

  const stats = {
    total: webhooks.length,
    active: webhooks.filter((w) => w.status === "active").length,
    failing: webhooks.filter((w) => w.status === "failing").length,
    deliveries24h: webhooks.reduce((sum, w) => sum + w.totalDeliveries, 0),
    failed24h: webhooks.reduce((sum, w) => sum + w.failedDeliveries, 0),
  };

  const handleSaveWebhook = (data: Partial<Webhook>) => {
    if (editingWebhook) {
      updateMutation.mutate({ id: editingWebhook.id, data });
    } else {
      createMutation.mutate(data);
    }
  };

  const handleDeleteWebhook = (webhookId: string) => {
    if (confirm("Are you sure you want to delete this webhook?")) {
      deleteMutation.mutate(webhookId);
    }
  };

  const handleToggleStatus = (webhookId: string) => {
    const webhook = webhooks.find((w) => w.id === webhookId);
    if (webhook) {
      updateMutation.mutate({
        id: webhookId,
        data: { status: webhook.status === "active" ? "paused" : "active" },
      });
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Webhooks</h1>
            <p className="text-muted-foreground">
              Receive real-time notifications for voice AI events
            </p>
          </div>
          <Button onClick={() => setShowCreateDialog(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Add Webhook
          </Button>
        </div>

        {/* Stats */}
        <div className="grid gap-4 md:grid-cols-5">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/30">
                  <Webhook className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.total}</p>
                  <p className="text-sm text-muted-foreground">Total Webhooks</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/30">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.active}</p>
                  <p className="text-sm text-muted-foreground">Active</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/30">
                  <AlertTriangle className="h-5 w-5 text-red-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.failing}</p>
                  <p className="text-sm text-muted-foreground">Failing</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-100 dark:bg-purple-900/30">
                  <Activity className="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.deliveries24h.toLocaleString()}</p>
                  <p className="text-sm text-muted-foreground">Deliveries (24h)</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-yellow-100 dark:bg-yellow-900/30">
                  <Clock className="h-5 w-5 text-yellow-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">145ms</p>
                  <p className="text-sm text-muted-foreground">Avg Response</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <div className="flex items-center justify-between">
            <TabsList>
              <TabsTrigger value="webhooks" className="gap-2">
                <Webhook className="h-4 w-4" />
                Webhooks
              </TabsTrigger>
              <TabsTrigger value="logs" className="gap-2">
                <History className="h-4 w-4" />
                Delivery Logs
              </TabsTrigger>
              <TabsTrigger value="events" className="gap-2">
                <Zap className="h-4 w-4" />
                Event Reference
              </TabsTrigger>
            </TabsList>

            {activeTab === "webhooks" && (
              <div className="flex gap-2">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search webhooks..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9 w-64"
                  />
                </div>
              </div>
            )}

            {activeTab === "logs" && (
              <Select value={selectedWebhookForLogs} onValueChange={setSelectedWebhookForLogs}>
                <SelectTrigger className="w-48">
                  <SelectValue placeholder="All webhooks" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All webhooks</SelectItem>
                  {webhooks.map((w) => (
                    <SelectItem key={w.id} value={w.id}>
                      {w.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>

          <TabsContent value="webhooks" className="space-y-4 mt-4">
            {isLoading ? (
              Array.from({ length: 3 }).map((_, i) => (
                <Card key={i}>
                  <CardContent className="p-6">
                    <Skeleton className="h-32 w-full" />
                  </CardContent>
                </Card>
              ))
            ) : filteredWebhooks.length > 0 ? (
              filteredWebhooks.map((webhook) => (
                <WebhookCard
                  key={webhook.id}
                  webhook={webhook}
                  onEdit={() => {
                    setEditingWebhook(webhook);
                    setShowCreateDialog(true);
                  }}
                  onTest={() => setTestingWebhook(webhook)}
                  onDelete={() => handleDeleteWebhook(webhook.id)}
                  onToggleStatus={() => handleToggleStatus(webhook.id)}
                />
              ))
            ) : (
              <Card>
                <CardContent className="p-12 text-center">
                  <Webhook className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium mb-1">No webhooks found</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    {searchQuery
                      ? "No webhooks match your search"
                      : "Create your first webhook to receive event notifications"}
                  </p>
                  {!searchQuery && (
                    <Button onClick={() => setShowCreateDialog(true)}>
                      <Plus className="mr-2 h-4 w-4" />
                      Create Webhook
                    </Button>
                  )}
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="logs" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>Delivery Logs</CardTitle>
                <CardDescription>
                  View recent webhook delivery attempts and their status
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                {filteredLogs.length > 0 ? (
                  filteredLogs.map((log) => <DeliveryLogRow key={log.id} log={log} />)
                ) : (
                  <div className="p-12 text-center">
                    <History className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-medium mb-1">No logs found</h3>
                    <p className="text-sm text-muted-foreground">
                      Delivery logs will appear here when webhooks are triggered
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="events" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>Available Events</CardTitle>
                <CardDescription>
                  Events you can subscribe to with your webhooks
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {Array.from(new Set(availableEvents.map((e) => e.category))).map(
                    (category) => (
                      <div key={category}>
                        <h4 className="font-medium mb-3">{category}</h4>
                        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                          {availableEvents
                            .filter((e) => e.category === category)
                            .map((event) => (
                              <div
                                key={event.id}
                                className="rounded-lg border p-4 hover:bg-muted/50 transition-colors"
                              >
                                <div className="flex items-center gap-2 mb-2">
                                  <Zap className="h-4 w-4 text-primary" />
                                  <code className="text-sm font-medium">{event.id}</code>
                                </div>
                                <p className="text-sm font-medium">{event.label}</p>
                                <p className="text-xs text-muted-foreground mt-1">
                                  {event.description}
                                </p>
                              </div>
                            ))}
                        </div>
                      </div>
                    )
                  )}
                </div>

                {/* Sample Payload */}
                <div className="mt-8 pt-6 border-t">
                  <h4 className="font-medium mb-3">Sample Webhook Payload</h4>
                  <div className="relative">
                    <pre className="p-4 bg-muted rounded-lg text-sm font-mono overflow-auto">
{`{
  "id": "evt_1234567890",
  "event": "call.ended",
  "timestamp": "2024-01-14T10:30:00Z",
  "data": {
    "call_id": "call-abc123",
    "agent_id": "agent-xyz",
    "caller_number": "+15551234567",
    "duration": 245,
    "sentiment": "positive",
    "outcome": "resolved",
    "transcript_url": "https://api.bvrai.com/transcripts/..."
  }
}`}
                    </pre>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="absolute top-2 right-2"
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Create/Edit Dialog */}
      <CreateWebhookDialog
        open={showCreateDialog}
        onOpenChange={(open) => {
          setShowCreateDialog(open);
          if (!open) setEditingWebhook(null);
        }}
        webhook={editingWebhook}
        onSave={handleSaveWebhook}
      />

      {/* Test Dialog */}
      {testingWebhook && (
        <TestWebhookDialog
          open={!!testingWebhook}
          onOpenChange={(open) => !open && setTestingWebhook(null)}
          webhook={testingWebhook}
        />
      )}
    </DashboardLayout>
  );
}

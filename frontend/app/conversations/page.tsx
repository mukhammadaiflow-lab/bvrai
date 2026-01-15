"use client";

import React, { useState, useEffect } from "react";
import {
  MessageSquare,
  Phone,
  PhoneCall,
  PhoneOff,
  Clock,
  User,
  Bot,
  Search,
  Filter,
  RefreshCw,
  ChevronRight,
  ThumbsUp,
  ThumbsDown,
  AlertTriangle,
  Mic,
  Volume2,
  MoreHorizontal,
  Eye,
  Download,
  ExternalLink,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn, formatDuration, formatRelativeTime, formatNumber } from "@/lib/utils";

// Mock active conversations
const activeConversations = [
  {
    id: "active-1",
    callId: "call-123",
    agentName: "Sales Agent",
    customerPhone: "+1 (555) 123-4567",
    status: "in_progress",
    duration: 145,
    sentiment: 0.75,
    startedAt: new Date(Date.now() - 145000).toISOString(),
    lastActivity: "Customer asking about pricing",
    transcript: [
      { role: "assistant", content: "Hello! Thank you for calling. How can I help you today?", timestamp: "0:00" },
      { role: "user", content: "Hi, I'm interested in learning more about your enterprise plan.", timestamp: "0:05" },
      { role: "assistant", content: "I'd be happy to help you with that! Our enterprise plan offers unlimited agents, priority support, and custom integrations.", timestamp: "0:12" },
      { role: "user", content: "What's the pricing for that?", timestamp: "0:25" },
    ],
  },
  {
    id: "active-2",
    callId: "call-124",
    agentName: "Support Agent",
    customerPhone: "+1 (555) 987-6543",
    status: "in_progress",
    duration: 89,
    sentiment: 0.45,
    startedAt: new Date(Date.now() - 89000).toISOString(),
    lastActivity: "Troubleshooting integration issue",
    transcript: [
      { role: "assistant", content: "Support team here. What seems to be the issue?", timestamp: "0:00" },
      { role: "user", content: "My webhook isn't receiving any events.", timestamp: "0:08" },
      { role: "assistant", content: "Let me help you debug that. Can you confirm your webhook URL is publicly accessible?", timestamp: "0:15" },
    ],
  },
  {
    id: "active-3",
    callId: "call-125",
    agentName: "Booking Agent",
    customerPhone: "+1 (555) 456-7890",
    status: "in_progress",
    duration: 42,
    sentiment: 0.85,
    startedAt: new Date(Date.now() - 42000).toISOString(),
    lastActivity: "Scheduling appointment",
    transcript: [
      { role: "assistant", content: "Hi! I can help you schedule an appointment. What date works best for you?", timestamp: "0:00" },
      { role: "user", content: "Do you have anything available next Tuesday afternoon?", timestamp: "0:07" },
    ],
  },
];

// Mock recent conversations
const recentConversations = [
  {
    id: "recent-1",
    callId: "call-120",
    agentName: "Sales Agent",
    customerPhone: "+1 (555) 111-2222",
    status: "completed",
    duration: 312,
    sentiment: 0.92,
    outcome: "lead_qualified",
    startedAt: "2024-01-14T10:15:00Z",
    endedAt: "2024-01-14T10:20:12Z",
    summary: "Customer interested in professional plan. Requested demo.",
    messageCount: 24,
  },
  {
    id: "recent-2",
    callId: "call-119",
    agentName: "Support Agent",
    customerPhone: "+1 (555) 333-4444",
    status: "completed",
    duration: 187,
    sentiment: 0.68,
    outcome: "issue_resolved",
    startedAt: "2024-01-14T10:00:00Z",
    endedAt: "2024-01-14T10:03:07Z",
    summary: "Helped customer reset API key and update webhook URL.",
    messageCount: 15,
  },
  {
    id: "recent-3",
    callId: "call-118",
    agentName: "Sales Agent",
    customerPhone: "+1 (555) 555-6666",
    status: "transferred",
    duration: 245,
    sentiment: 0.55,
    outcome: "transferred_to_human",
    startedAt: "2024-01-14T09:45:00Z",
    endedAt: "2024-01-14T09:49:05Z",
    summary: "Complex enterprise requirements. Transferred to sales team.",
    messageCount: 18,
  },
  {
    id: "recent-4",
    callId: "call-117",
    agentName: "Booking Agent",
    customerPhone: "+1 (555) 777-8888",
    status: "completed",
    duration: 156,
    sentiment: 0.88,
    outcome: "appointment_booked",
    startedAt: "2024-01-14T09:30:00Z",
    endedAt: "2024-01-14T09:32:36Z",
    summary: "Booked appointment for January 18th at 2:00 PM.",
    messageCount: 12,
  },
  {
    id: "recent-5",
    callId: "call-116",
    agentName: "Support Agent",
    customerPhone: "+1 (555) 999-0000",
    status: "failed",
    duration: 23,
    sentiment: 0.25,
    outcome: "call_dropped",
    startedAt: "2024-01-14T09:15:00Z",
    endedAt: "2024-01-14T09:15:23Z",
    summary: "Call dropped due to network issues.",
    messageCount: 3,
  },
];

export default function ConversationsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedConversation, setSelectedConversation] = useState<string | null>("active-1");
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(Date.now());

  // Update current time every second for live duration display
  useEffect(() => {
    const interval = setInterval(() => setCurrentTime(Date.now()), 1000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "in_progress":
        return "bg-green-100 text-green-800 border-green-200";
      case "completed":
        return "bg-gray-100 text-gray-800 border-gray-200";
      case "transferred":
        return "bg-blue-100 text-blue-800 border-blue-200";
      case "failed":
        return "bg-red-100 text-red-800 border-red-200";
      default:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getSentimentColor = (sentiment: number) => {
    if (sentiment >= 0.7) return "text-green-600";
    if (sentiment >= 0.4) return "text-yellow-600";
    return "text-red-600";
  };

  const getSentimentIcon = (sentiment: number) => {
    if (sentiment >= 0.7) return <ThumbsUp className="h-4 w-4" />;
    if (sentiment >= 0.4) return <AlertTriangle className="h-4 w-4" />;
    return <ThumbsDown className="h-4 w-4" />;
  };

  const selectedActive = activeConversations.find((c) => c.id === selectedConversation);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Conversations</h1>
          <p className="text-muted-foreground">
            Monitor active calls and review conversation history
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm">
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
          <Button variant="outline" size="sm">
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Active Calls Summary */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-100">
                <PhoneCall className="h-5 w-5 text-green-600 animate-pulse" />
              </div>
              <div>
                <p className="text-2xl font-bold">{activeConversations.length}</p>
                <p className="text-sm text-muted-foreground">Active Calls</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                <MessageSquare className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">156</p>
                <p className="text-sm text-muted-foreground">Today's Calls</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-yellow-100">
                <Clock className="h-5 w-5 text-yellow-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">2:45</p>
                <p className="text-sm text-muted-foreground">Avg Duration</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-100">
                <ThumbsUp className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">78%</p>
                <p className="text-sm text-muted-foreground">Positive Sentiment</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Conversations List */}
        <div className="lg:col-span-1 space-y-4">
          {/* Search and Filters */}
          <div className="flex gap-2">
            <Input
              placeholder="Search conversations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              leftIcon={<Search className="h-4 w-4" />}
              className="flex-1"
            />
            <Button variant="outline" size="icon">
              <Filter className="h-4 w-4" />
            </Button>
          </div>

          {/* Active Conversations */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                Live Calls ({activeConversations.length})
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {activeConversations.map((conv) => {
                const liveDuration = Math.floor((currentTime - new Date(conv.startedAt).getTime()) / 1000);
                return (
                  <div
                    key={conv.id}
                    onClick={() => setSelectedConversation(conv.id)}
                    className={cn(
                      "rounded-lg border p-3 cursor-pointer transition-all",
                      selectedConversation === conv.id
                        ? "border-primary bg-primary/5"
                        : "hover:bg-muted/50"
                    )}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <PhoneCall className="h-4 w-4 text-green-600 animate-pulse" />
                        <span className="font-medium text-sm">{conv.customerPhone}</span>
                      </div>
                      <Badge className="bg-green-100 text-green-800 text-xs">
                        {formatDuration(liveDuration)}
                      </Badge>
                    </div>
                    <p className="text-xs text-muted-foreground mb-1">{conv.agentName}</p>
                    <p className="text-xs truncate">{conv.lastActivity}</p>
                    <div className="flex items-center gap-2 mt-2">
                      <div className={cn("flex items-center gap-1 text-xs", getSentimentColor(conv.sentiment))}>
                        {getSentimentIcon(conv.sentiment)}
                        {(conv.sentiment * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                );
              })}
            </CardContent>
          </Card>

          {/* Recent Conversations */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Recent Conversations</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {recentConversations.map((conv) => (
                <div
                  key={conv.id}
                  onClick={() => setSelectedConversation(conv.id)}
                  className={cn(
                    "rounded-lg border p-3 cursor-pointer transition-all",
                    selectedConversation === conv.id
                      ? "border-primary bg-primary/5"
                      : "hover:bg-muted/50"
                  )}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {conv.status === "completed" ? (
                        <Phone className="h-4 w-4 text-gray-600" />
                      ) : conv.status === "transferred" ? (
                        <ExternalLink className="h-4 w-4 text-blue-600" />
                      ) : (
                        <PhoneOff className="h-4 w-4 text-red-600" />
                      )}
                      <span className="font-medium text-sm">{conv.customerPhone}</span>
                    </div>
                    <Badge className={cn("text-xs", getStatusColor(conv.status))}>
                      {conv.status}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground mb-1">{conv.agentName}</p>
                  <p className="text-xs truncate">{conv.summary}</p>
                  <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
                    <span>{formatDuration(conv.duration)}</span>
                    <span>{formatRelativeTime(conv.endedAt!)}</span>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Conversation Detail */}
        <div className="lg:col-span-2">
          {selectedActive ? (
            <Card className="h-full">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <Badge className="bg-green-100 text-green-800">
                        <PhoneCall className="mr-1 h-3 w-3 animate-pulse" />
                        Live Call
                      </Badge>
                      <span className="text-lg font-medium">{selectedActive.customerPhone}</span>
                    </div>
                    <CardDescription>
                      Agent: {selectedActive.agentName} | Call ID: {selectedActive.callId}
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm">
                      <Eye className="mr-1 h-4 w-4" />
                      Monitor
                    </Button>
                    <Button variant="destructive" size="sm">
                      <PhoneOff className="mr-1 h-4 w-4" />
                      End Call
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {/* Call Stats */}
                <div className="grid grid-cols-4 gap-4 mb-6">
                  <div className="rounded-lg border p-3 text-center">
                    <Clock className="h-5 w-5 mx-auto mb-1 text-muted-foreground" />
                    <p className="text-lg font-bold">
                      {formatDuration(Math.floor((currentTime - new Date(selectedActive.startedAt).getTime()) / 1000))}
                    </p>
                    <p className="text-xs text-muted-foreground">Duration</p>
                  </div>
                  <div className="rounded-lg border p-3 text-center">
                    <MessageSquare className="h-5 w-5 mx-auto mb-1 text-muted-foreground" />
                    <p className="text-lg font-bold">{selectedActive.transcript.length}</p>
                    <p className="text-xs text-muted-foreground">Messages</p>
                  </div>
                  <div className="rounded-lg border p-3 text-center">
                    <div className={cn("mx-auto mb-1", getSentimentColor(selectedActive.sentiment))}>
                      {getSentimentIcon(selectedActive.sentiment)}
                    </div>
                    <p className="text-lg font-bold">{(selectedActive.sentiment * 100).toFixed(0)}%</p>
                    <p className="text-xs text-muted-foreground">Sentiment</p>
                  </div>
                  <div className="rounded-lg border p-3 text-center">
                    <Mic className="h-5 w-5 mx-auto mb-1 text-green-500" />
                    <p className="text-lg font-bold">Active</p>
                    <p className="text-xs text-muted-foreground">Listening</p>
                  </div>
                </div>

                {/* Live Transcript */}
                <div>
                  <h4 className="font-medium mb-3 flex items-center gap-2">
                    <MessageSquare className="h-4 w-4" />
                    Live Transcript
                  </h4>
                  <div className="rounded-lg border bg-muted/30 p-4 max-h-[400px] overflow-y-auto space-y-4">
                    {selectedActive.transcript.map((msg, index) => (
                      <div
                        key={index}
                        className={cn(
                          "flex gap-3",
                          msg.role === "assistant" ? "flex-row" : "flex-row-reverse"
                        )}
                      >
                        <div className={cn(
                          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
                          msg.role === "assistant" ? "bg-primary text-primary-foreground" : "bg-secondary"
                        )}>
                          {msg.role === "assistant" ? (
                            <Bot className="h-4 w-4" />
                          ) : (
                            <User className="h-4 w-4" />
                          )}
                        </div>
                        <div className={cn(
                          "rounded-lg p-3 max-w-[80%]",
                          msg.role === "assistant"
                            ? "bg-primary/10"
                            : "bg-secondary"
                        )}>
                          <p className="text-sm">{msg.content}</p>
                          <p className="text-xs text-muted-foreground mt-1">{msg.timestamp}</p>
                        </div>
                      </div>
                    ))}
                    {/* Typing indicator */}
                    <div className="flex gap-3">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                        <Bot className="h-4 w-4" />
                      </div>
                      <div className="rounded-lg bg-primary/10 p-3">
                        <div className="flex gap-1">
                          <div className="h-2 w-2 rounded-full bg-primary/50 animate-bounce" style={{ animationDelay: "0ms" }} />
                          <div className="h-2 w-2 rounded-full bg-primary/50 animate-bounce" style={{ animationDelay: "150ms" }} />
                          <div className="h-2 w-2 rounded-full bg-primary/50 animate-bounce" style={{ animationDelay: "300ms" }} />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Audio Indicators */}
                <div className="flex items-center justify-between mt-4 p-3 rounded-lg border">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <Mic className="h-4 w-4 text-green-500" />
                      <span className="text-sm">Customer Audio</span>
                      <div className="flex gap-0.5">
                        {[...Array(5)].map((_, i) => (
                          <div
                            key={i}
                            className={cn(
                              "w-1 rounded-full bg-green-500 transition-all",
                              i < 3 ? "h-3" : i < 4 ? "h-2" : "h-1"
                            )}
                            style={{
                              opacity: Math.random() > 0.3 ? 1 : 0.3,
                            }}
                          />
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Volume2 className="h-4 w-4 text-blue-500" />
                      <span className="text-sm">Agent Audio</span>
                      <div className="flex gap-0.5">
                        {[...Array(5)].map((_, i) => (
                          <div
                            key={i}
                            className="w-1 h-1 rounded-full bg-blue-500"
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                  <Button variant="ghost" size="sm">
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="h-full flex items-center justify-center">
              <CardContent className="text-center py-12">
                <MessageSquare className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">Select a conversation</h3>
                <p className="text-muted-foreground">
                  Choose a conversation from the list to view details
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

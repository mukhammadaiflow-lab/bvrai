"use client";

import React, { useState, useMemo } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import {
  Phone,
  PhoneIncoming,
  PhoneOutgoing,
  PhoneOff,
  Bot,
  Clock,
  Calendar,
  User,
  MessageSquare,
  Wrench,
  TrendingUp,
  TrendingDown,
  Minus,
  CheckCircle2,
  XCircle,
  AlertCircle,
  ArrowLeft,
  Download,
  Share2,
  Flag,
  Star,
  MoreHorizontal,
  Copy,
  ExternalLink,
  RefreshCw,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress, CircularProgress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AudioPlayer,
  TranscriptViewer,
  TranscriptMessage,
  CallTimeline,
  CallTimelineCompact,
  generateTimelineEvents,
} from "@/components/calls";
import { cn, formatDate, formatDuration, formatPhoneNumber, getStatusColor } from "@/lib/utils";
import { calls } from "@/lib/api";
import { toast } from "sonner";

// Skeleton component
function CallDetailSkeleton() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Skeleton className="h-8 w-8 rounded" />
        <Skeleton className="h-6 w-32" />
      </div>
      <div className="flex items-start gap-6">
        <Skeleton className="h-16 w-16 rounded-xl" />
        <div className="space-y-2">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-4 w-64" />
          <div className="flex gap-2 pt-2">
            <Skeleton className="h-6 w-20 rounded-full" />
            <Skeleton className="h-6 w-16 rounded-full" />
          </div>
        </div>
      </div>
      <div className="grid grid-cols-4 gap-4">
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
      </div>
    </div>
  );
}

export default function CallDetailPage() {
  const params = useParams();
  const router = useRouter();
  const callId = params.id as string;

  const [activeTab, setActiveTab] = useState("overview");
  const [playbackTime, setPlaybackTime] = useState(0);

  // Fetch call details
  const { data: call, isLoading, error, refetch } = useQuery({
    queryKey: ["call", callId],
    queryFn: () => calls.get(callId),
    enabled: !!callId,
    refetchInterval: (data) => {
      // Auto-refresh if call is in progress
      return data?.status === "in_progress" ? 5000 : false;
    },
  });

  // Parse transcript into messages
  const transcriptMessages: TranscriptMessage[] = useMemo(() => {
    if (!call?.transcript) return [];
    return call.transcript.map((msg: any, index: number) => ({
      id: `msg-${index}`,
      role: msg.role || (msg.speaker === "agent" ? "agent" : "user"),
      content: msg.content || msg.text || "",
      timestamp: msg.timestamp || msg.start_time || index * 5,
      sentiment: msg.sentiment,
      confidence: msg.confidence,
      tool_use: msg.tool_use,
    }));
  }, [call?.transcript]);

  // Generate timeline events
  const timelineEvents = useMemo(() => {
    if (!call) return [];
    return generateTimelineEvents(call);
  }, [call]);

  // Copy call ID
  const handleCopyId = async () => {
    await navigator.clipboard.writeText(callId);
    toast.success("Call ID copied to clipboard");
  };

  // Error state
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <AlertCircle className="h-12 w-12 text-destructive" />
        <p className="text-lg font-medium">Failed to load call</p>
        <p className="text-sm text-muted-foreground">
          {error instanceof Error ? error.message : "Call not found"}
        </p>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/calls">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Calls
            </Link>
          </Button>
          <Button onClick={() => refetch()}>Retry</Button>
        </div>
      </div>
    );
  }

  // Loading state
  if (isLoading) {
    return <CallDetailSkeleton />;
  }

  const isInbound = call?.direction === "inbound";
  const duration = call?.duration_seconds || call?.duration || 0;

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Link href="/calls" className="hover:text-foreground transition-colors">
          Calls
        </Link>
        <span>/</span>
        <span className="text-foreground font-mono text-xs">
          {callId.slice(0, 8)}...
        </span>
      </div>

      {/* Header */}
      <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
        <div className="flex items-start gap-5">
          <div
            className={cn(
              "flex h-16 w-16 items-center justify-center rounded-xl",
              call?.status === "completed"
                ? "bg-green-100 dark:bg-green-950"
                : call?.status === "failed"
                ? "bg-red-100 dark:bg-red-950"
                : call?.status === "in_progress"
                ? "bg-blue-100 dark:bg-blue-950"
                : "bg-gray-100 dark:bg-gray-900"
            )}
          >
            {call?.status === "completed" ? (
              <CheckCircle2 className="h-8 w-8 text-green-600 dark:text-green-400" />
            ) : call?.status === "failed" ? (
              <XCircle className="h-8 w-8 text-red-600 dark:text-red-400" />
            ) : call?.status === "in_progress" ? (
              <Phone className="h-8 w-8 text-blue-600 dark:text-blue-400 animate-pulse" />
            ) : (
              <PhoneOff className="h-8 w-8 text-gray-600 dark:text-gray-400" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold">
                {formatPhoneNumber(isInbound ? call?.from_number : call?.to_number)}
              </h1>
              <Badge className={getStatusColor(call?.status)}>
                {call?.status?.replace("_", " ")}
              </Badge>
              {call?.status === "in_progress" && (
                <span className="flex items-center gap-1 text-sm text-green-600">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                  </span>
                  Live
                </span>
              )}
            </div>
            <p className="text-muted-foreground mt-1">
              {isInbound ? "Inbound call" : "Outbound call"} with{" "}
              <Link href={`/agents/${call?.agent_id}`} className="text-primary hover:underline">
                {call?.agent_name || "Voice Agent"}
              </Link>
            </p>
            <div className="flex flex-wrap items-center gap-3 mt-3">
              <Badge variant="secondary" className="font-normal">
                {isInbound ? (
                  <PhoneIncoming className="mr-1 h-3 w-3" />
                ) : (
                  <PhoneOutgoing className="mr-1 h-3 w-3" />
                )}
                {isInbound ? "Inbound" : "Outbound"}
              </Badge>
              <Badge variant="secondary" className="font-normal">
                <Clock className="mr-1 h-3 w-3" />
                {formatDuration(duration)}
              </Badge>
              <Badge variant="secondary" className="font-normal">
                <Calendar className="mr-1 h-3 w-3" />
                {formatDate(call?.created_at || call?.initiated_at, "PPp")}
              </Badge>
              {call?.sentiment && (
                <Badge
                  variant={
                    call.sentiment === "positive"
                      ? "success"
                      : call.sentiment === "negative"
                      ? "destructive"
                      : "secondary"
                  }
                  className="font-normal"
                >
                  {call.sentiment === "positive" ? (
                    <TrendingUp className="mr-1 h-3 w-3" />
                  ) : call.sentiment === "negative" ? (
                    <TrendingDown className="mr-1 h-3 w-3" />
                  ) : (
                    <Minus className="mr-1 h-3 w-3" />
                  )}
                  {call.sentiment}
                </Badge>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {call?.status === "in_progress" && (
            <Button variant="destructive">
              <PhoneOff className="mr-2 h-4 w-4" />
              End Call
            </Button>
          )}
          <Button variant="outline" onClick={() => refetch()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="icon">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={handleCopyId}>
                <Copy className="mr-2 h-4 w-4" />
                Copy Call ID
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Share2 className="mr-2 h-4 w-4" />
                Share
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Download className="mr-2 h-4 w-4" />
                Export
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <Flag className="mr-2 h-4 w-4" />
                Flag for Review
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Star className="mr-2 h-4 w-4" />
                Add to Favorites
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Audio Player */}
      {call?.recording_url && (
        <AudioPlayer
          src={call.recording_url}
          title={`Call Recording - ${formatPhoneNumber(isInbound ? call.from_number : call.to_number)}`}
          onTimeUpdate={setPlaybackTime}
        />
      )}

      {/* Timeline (Compact) */}
      {timelineEvents.length > 0 && (
        <CallTimelineCompact
          events={timelineEvents}
          currentTime={playbackTime}
          duration={duration}
        />
      )}

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Clock className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Duration</p>
                <p className="text-xl font-bold">{formatDuration(duration)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <MessageSquare className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Messages</p>
                <p className="text-xl font-bold">{transcriptMessages.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Wrench className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Tool Uses</p>
                <p className="text-xl font-bold">
                  {transcriptMessages.filter((m) => m.tool_use).length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <TrendingUp className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Confidence</p>
                <p className="text-xl font-bold">
                  {call?.confidence ? `${(call.confidence * 100).toFixed(0)}%` : "N/A"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="transcript">Transcript</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6 mt-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Summary */}
            <Card>
              <CardHeader>
                <CardTitle>Call Summary</CardTitle>
              </CardHeader>
              <CardContent>
                {call?.summary ? (
                  <p className="text-sm">{call.summary}</p>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No summary available for this call.
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Key Points */}
            <Card>
              <CardHeader>
                <CardTitle>Key Points</CardTitle>
              </CardHeader>
              <CardContent>
                {call?.key_points?.length > 0 ? (
                  <ul className="space-y-2">
                    {call.key_points.map((point: string, index: number) => (
                      <li key={index} className="flex items-start gap-2 text-sm">
                        <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No key points extracted from this call.
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Call Details */}
            <Card>
              <CardHeader>
                <CardTitle>Call Details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">From</p>
                    <p className="font-medium">{formatPhoneNumber(call?.from_number)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">To</p>
                    <p className="font-medium">{formatPhoneNumber(call?.to_number)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Started</p>
                    <p className="font-medium">
                      {formatDate(call?.created_at || call?.initiated_at, "PPp")}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Ended</p>
                    <p className="font-medium">
                      {call?.ended_at ? formatDate(call.ended_at, "PPp") : "N/A"}
                    </p>
                  </div>
                </div>
                <Separator />
                <div>
                  <p className="text-sm text-muted-foreground">End Reason</p>
                  <p className="font-medium">{call?.end_reason || "Call completed normally"}</p>
                </div>
              </CardContent>
            </Card>

            {/* Agent Info */}
            <Card>
              <CardHeader>
                <CardTitle>Agent Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                    <Bot className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <p className="font-medium">{call?.agent_name || "Voice Agent"}</p>
                    <p className="text-sm text-muted-foreground">
                      {call?.agent_id ? (
                        <Link href={`/agents/${call.agent_id}`} className="text-primary hover:underline">
                          View Agent
                        </Link>
                      ) : (
                        "Agent details not available"
                      )}
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Model</p>
                    <p className="font-medium">{call?.llm_model || "GPT-4o"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Voice</p>
                    <p className="font-medium">{call?.voice_name || "Default"}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Transcript Tab */}
        <TabsContent value="transcript" className="mt-6">
          <TranscriptViewer
            messages={transcriptMessages}
            currentTime={playbackTime}
            onMessageClick={(timestamp) => setPlaybackTime(timestamp)}
          />
        </TabsContent>

        {/* Timeline Tab */}
        <TabsContent value="timeline" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Call Timeline</CardTitle>
              <CardDescription>
                Events and milestones throughout the call
              </CardDescription>
            </CardHeader>
            <CardContent>
              <CallTimeline
                events={timelineEvents}
                currentTime={playbackTime}
                duration={duration}
              />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis" className="space-y-6 mt-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Sentiment Analysis */}
            <Card>
              <CardHeader>
                <CardTitle>Sentiment Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Overall Sentiment</span>
                    <Badge
                      variant={
                        call?.sentiment === "positive"
                          ? "success"
                          : call?.sentiment === "negative"
                          ? "destructive"
                          : "secondary"
                      }
                    >
                      {call?.sentiment || "neutral"}
                    </Badge>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Positive</span>
                      <span className="text-green-600">{call?.sentiment_scores?.positive || 30}%</span>
                    </div>
                    <Progress value={call?.sentiment_scores?.positive || 30} className="h-2" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Neutral</span>
                      <span>{call?.sentiment_scores?.neutral || 50}%</span>
                    </div>
                    <Progress value={call?.sentiment_scores?.neutral || 50} className="h-2" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Negative</span>
                      <span className="text-red-600">{call?.sentiment_scores?.negative || 20}%</span>
                    </div>
                    <Progress value={call?.sentiment_scores?.negative || 20} className="h-2 [&>div]:bg-red-500" />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Topics Discussed */}
            <Card>
              <CardHeader>
                <CardTitle>Topics Discussed</CardTitle>
              </CardHeader>
              <CardContent>
                {call?.topics?.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {call.topics.map((topic: string, index: number) => (
                      <Badge key={index} variant="secondary">
                        {topic}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No topics identified for this call.
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Intent Detection */}
            <Card>
              <CardHeader>
                <CardTitle>Detected Intents</CardTitle>
              </CardHeader>
              <CardContent>
                {call?.intents?.length > 0 ? (
                  <div className="space-y-2">
                    {call.intents.map((intent: any, index: number) => (
                      <div key={index} className="flex items-center justify-between p-2 rounded bg-muted/50">
                        <span className="text-sm font-medium">{intent.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {(intent.confidence * 100).toFixed(0)}% confidence
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No intents detected for this call.
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Action Items */}
            <Card>
              <CardHeader>
                <CardTitle>Action Items</CardTitle>
              </CardHeader>
              <CardContent>
                {call?.action_items?.length > 0 ? (
                  <ul className="space-y-2">
                    {call.action_items.map((item: string, index: number) => (
                      <li key={index} className="flex items-start gap-2 text-sm">
                        <div className="h-5 w-5 rounded border flex items-center justify-center shrink-0 mt-0.5">
                          <span className="text-xs text-muted-foreground">{index + 1}</span>
                        </div>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No action items identified from this call.
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {/* Metadata */}
      <Card>
        <CardHeader>
          <CardTitle>Technical Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">Call ID</p>
              <p className="font-mono text-xs">{callId}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Provider</p>
              <p>{call?.provider || "Twilio"}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Region</p>
              <p>{call?.region || "us-east-1"}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Cost</p>
              <p>{call?.cost ? `$${call.cost.toFixed(4)}` : "N/A"}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

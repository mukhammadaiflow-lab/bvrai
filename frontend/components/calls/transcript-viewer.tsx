"use client";

import React, { useRef, useEffect, useState } from "react";
import { Bot, User, Copy, Check, Search, Download, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn, formatDuration } from "@/lib/utils";
import { toast } from "sonner";

export interface TranscriptMessage {
  id: string;
  role: "agent" | "user" | "system";
  content: string;
  timestamp: number; // in seconds
  sentiment?: "positive" | "neutral" | "negative";
  confidence?: number;
  tool_use?: {
    name: string;
    result?: string;
  };
}

interface TranscriptViewerProps {
  messages: TranscriptMessage[];
  currentTime?: number;
  onMessageClick?: (timestamp: number) => void;
  showTimestamps?: boolean;
  className?: string;
}

export function TranscriptViewer({
  messages,
  currentTime = 0,
  onMessageClick,
  showTimestamps = true,
  className,
}: TranscriptViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [copied, setCopied] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);

  // Find the current message based on playback time
  const currentMessageIndex = messages.findIndex((msg, index) => {
    const nextMsg = messages[index + 1];
    return currentTime >= msg.timestamp && (!nextMsg || currentTime < nextMsg.timestamp);
  });

  // Auto-scroll to current message
  useEffect(() => {
    if (autoScroll && containerRef.current && currentMessageIndex >= 0) {
      const element = containerRef.current.children[currentMessageIndex] as HTMLElement;
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }, [currentMessageIndex, autoScroll]);

  // Filter messages by search
  const filteredMessages = searchQuery
    ? messages.filter((msg) =>
        msg.content.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : messages;

  // Copy transcript to clipboard
  const handleCopy = async () => {
    const text = messages
      .map((msg) => `[${formatDuration(msg.timestamp)}] ${msg.role === "agent" ? "Agent" : "User"}: ${msg.content}`)
      .join("\n\n");

    await navigator.clipboard.writeText(text);
    setCopied(true);
    toast.success("Transcript copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  // Export transcript
  const handleExport = () => {
    const text = messages
      .map((msg) => `[${formatDuration(msg.timestamp)}] ${msg.role === "agent" ? "Agent" : "User"}: ${msg.content}`)
      .join("\n\n");

    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "transcript.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (messages.length === 0) {
    return (
      <div className={cn("rounded-lg border bg-muted/50 p-8 text-center", className)}>
        <Bot className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">No transcript available</p>
      </div>
    );
  }

  return (
    <div className={cn("rounded-lg border bg-card", className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-4">
          <h3 className="font-medium">Conversation Transcript</h3>
          <Badge variant="secondary">{messages.length} messages</Badge>
        </div>
        <div className="flex items-center gap-2">
          <Input
            placeholder="Search transcript..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            leftIcon={<Search className="h-4 w-4" />}
            className="w-48 h-8 text-sm"
          />
          <Button variant="ghost" size="icon-sm" onClick={handleCopy} title="Copy transcript">
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          </Button>
          <Button variant="ghost" size="icon-sm" onClick={handleExport} title="Download transcript">
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={containerRef}
        className="p-4 space-y-4 max-h-[500px] overflow-y-auto"
        onMouseEnter={() => setAutoScroll(false)}
        onMouseLeave={() => setAutoScroll(true)}
      >
        {filteredMessages.map((message, index) => {
          const isActive = index === currentMessageIndex && !searchQuery;
          const isAgent = message.role === "agent";
          const isSystem = message.role === "system";

          return (
            <div
              key={message.id}
              className={cn(
                "flex gap-3 p-3 rounded-lg transition-all cursor-pointer",
                isActive && "bg-primary/10 ring-1 ring-primary/30",
                !isActive && "hover:bg-muted/50",
                isSystem && "bg-muted/30"
              )}
              onClick={() => onMessageClick?.(message.timestamp)}
            >
              {/* Avatar */}
              <div
                className={cn(
                  "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
                  isAgent
                    ? "bg-primary/10 text-primary"
                    : isSystem
                    ? "bg-muted text-muted-foreground"
                    : "bg-secondary text-secondary-foreground"
                )}
              >
                {isAgent ? (
                  <Bot className="h-4 w-4" />
                ) : isSystem ? (
                  <Clock className="h-4 w-4" />
                ) : (
                  <User className="h-4 w-4" />
                )}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-medium text-sm">
                    {isAgent ? "Agent" : isSystem ? "System" : "User"}
                  </span>
                  {showTimestamps && (
                    <span className="text-xs text-muted-foreground">
                      {formatDuration(message.timestamp)}
                    </span>
                  )}
                  {message.sentiment && (
                    <Badge
                      variant={
                        message.sentiment === "positive"
                          ? "success"
                          : message.sentiment === "negative"
                          ? "destructive"
                          : "secondary"
                      }
                      className="text-xs py-0"
                    >
                      {message.sentiment}
                    </Badge>
                  )}
                </div>
                <p className={cn(
                  "text-sm whitespace-pre-wrap",
                  searchQuery && message.content.toLowerCase().includes(searchQuery.toLowerCase())
                    ? "bg-yellow-100 dark:bg-yellow-900/30"
                    : ""
                )}>
                  {message.content}
                </p>

                {/* Tool Use */}
                {message.tool_use && (
                  <div className="mt-2 p-2 rounded bg-muted/50 text-xs">
                    <span className="font-mono text-primary">{message.tool_use.name}</span>
                    {message.tool_use.result && (
                      <span className="text-muted-foreground ml-2">
                        {message.tool_use.result}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer with stats */}
      <div className="flex items-center justify-between px-4 py-2 border-t bg-muted/30 text-xs text-muted-foreground">
        <span>
          {messages.filter((m) => m.role === "agent").length} agent messages,{" "}
          {messages.filter((m) => m.role === "user").length} user messages
        </span>
        {currentMessageIndex >= 0 && !searchQuery && (
          <span>
            Currently at message {currentMessageIndex + 1} of {messages.length}
          </span>
        )}
      </div>
    </div>
  );
}

// Compact transcript view for previews
export function TranscriptPreview({
  messages,
  maxMessages = 3,
  className,
}: {
  messages: TranscriptMessage[];
  maxMessages?: number;
  className?: string;
}) {
  const previewMessages = messages.slice(0, maxMessages);

  return (
    <div className={cn("space-y-2", className)}>
      {previewMessages.map((message) => (
        <div key={message.id} className="flex gap-2 text-sm">
          <span className={cn(
            "font-medium shrink-0",
            message.role === "agent" ? "text-primary" : "text-muted-foreground"
          )}>
            {message.role === "agent" ? "Agent:" : "User:"}
          </span>
          <span className="text-muted-foreground line-clamp-1">{message.content}</span>
        </div>
      ))}
      {messages.length > maxMessages && (
        <p className="text-xs text-muted-foreground">
          +{messages.length - maxMessages} more messages
        </p>
      )}
    </div>
  );
}

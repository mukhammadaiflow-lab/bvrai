"use client";

import React from "react";
import {
  PhoneIncoming,
  PhoneOutgoing,
  Bot,
  MessageSquare,
  Wrench,
  CheckCircle2,
  XCircle,
  PhoneOff,
  Clock,
  AlertTriangle,
} from "lucide-react";
import { cn, formatDuration } from "@/lib/utils";

export interface TimelineEvent {
  id: string;
  type: "call_started" | "agent_connected" | "message" | "tool_use" | "transfer" | "call_ended" | "error";
  timestamp: number; // in seconds from start
  title: string;
  description?: string;
  metadata?: Record<string, any>;
}

interface CallTimelineProps {
  events: TimelineEvent[];
  currentTime?: number;
  duration: number;
  className?: string;
}

const eventIcons: Record<TimelineEvent["type"], React.ElementType> = {
  call_started: PhoneIncoming,
  agent_connected: Bot,
  message: MessageSquare,
  tool_use: Wrench,
  transfer: PhoneOutgoing,
  call_ended: PhoneOff,
  error: AlertTriangle,
};

const eventColors: Record<TimelineEvent["type"], string> = {
  call_started: "bg-blue-100 text-blue-600 border-blue-200",
  agent_connected: "bg-purple-100 text-purple-600 border-purple-200",
  message: "bg-gray-100 text-gray-600 border-gray-200",
  tool_use: "bg-amber-100 text-amber-600 border-amber-200",
  transfer: "bg-green-100 text-green-600 border-green-200",
  call_ended: "bg-slate-100 text-slate-600 border-slate-200",
  error: "bg-red-100 text-red-600 border-red-200",
};

export function CallTimeline({
  events,
  currentTime = 0,
  duration,
  className,
}: CallTimelineProps) {
  return (
    <div className={cn("relative", className)}>
      {/* Timeline line */}
      <div className="absolute left-5 top-0 bottom-0 w-0.5 bg-border" />

      {/* Progress indicator */}
      {currentTime > 0 && duration > 0 && (
        <div
          className="absolute left-5 top-0 w-0.5 bg-primary transition-all duration-300"
          style={{ height: `${(currentTime / duration) * 100}%` }}
        />
      )}

      {/* Events */}
      <div className="space-y-4">
        {events.map((event, index) => {
          const Icon = eventIcons[event.type];
          const isActive = currentTime >= event.timestamp;
          const isCurrent = index === events.findIndex((e, i) => {
            const nextEvent = events[i + 1];
            return currentTime >= e.timestamp && (!nextEvent || currentTime < nextEvent.timestamp);
          });

          return (
            <div key={event.id} className="relative flex gap-4 pl-2">
              {/* Icon */}
              <div
                className={cn(
                  "flex h-8 w-8 shrink-0 items-center justify-center rounded-full border-2 z-10 transition-all",
                  eventColors[event.type],
                  !isActive && "opacity-50",
                  isCurrent && "ring-2 ring-primary ring-offset-2"
                )}
              >
                <Icon className="h-4 w-4" />
              </div>

              {/* Content */}
              <div className={cn("flex-1 min-w-0 pb-4", !isActive && "opacity-50")}>
                <div className="flex items-center gap-2">
                  <span className="font-medium text-sm">{event.title}</span>
                  <span className="text-xs text-muted-foreground">
                    {formatDuration(event.timestamp)}
                  </span>
                </div>
                {event.description && (
                  <p className="text-sm text-muted-foreground mt-0.5">
                    {event.description}
                  </p>
                )}
                {event.metadata && Object.keys(event.metadata).length > 0 && (
                  <div className="mt-2 p-2 rounded bg-muted/50 text-xs font-mono">
                    {Object.entries(event.metadata).map(([key, value]) => (
                      <div key={key} className="flex gap-2">
                        <span className="text-muted-foreground">{key}:</span>
                        <span>{String(value)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Compact horizontal timeline
export function CallTimelineCompact({
  events,
  currentTime = 0,
  duration,
  className,
}: CallTimelineProps) {
  return (
    <div className={cn("relative", className)}>
      {/* Progress bar */}
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${(currentTime / duration) * 100}%` }}
        />
      </div>

      {/* Event markers */}
      <div className="relative h-6 mt-1">
        {events.map((event) => {
          const position = (event.timestamp / duration) * 100;
          const Icon = eventIcons[event.type];

          return (
            <div
              key={event.id}
              className="absolute -translate-x-1/2 group"
              style={{ left: `${position}%` }}
            >
              <div
                className={cn(
                  "flex h-5 w-5 items-center justify-center rounded-full border",
                  eventColors[event.type],
                  "cursor-pointer hover:scale-125 transition-transform"
                )}
              >
                <Icon className="h-3 w-3" />
              </div>

              {/* Tooltip */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-10">
                <div className="bg-popover text-popover-foreground rounded-md shadow-md px-2 py-1 text-xs whitespace-nowrap">
                  <p className="font-medium">{event.title}</p>
                  <p className="text-muted-foreground">{formatDuration(event.timestamp)}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Generate timeline events from call data
export function generateTimelineEvents(call: any): TimelineEvent[] {
  const events: TimelineEvent[] = [];

  // Call started
  events.push({
    id: "start",
    type: "call_started",
    timestamp: 0,
    title: "Call Started",
    description: call.direction === "inbound"
      ? `Incoming call from ${call.from_number}`
      : `Outgoing call to ${call.to_number}`,
  });

  // Agent connected
  events.push({
    id: "agent",
    type: "agent_connected",
    timestamp: 1,
    title: "Agent Connected",
    description: `${call.agent_name || "Voice Agent"} joined the call`,
  });

  // Add transcript events
  if (call.transcript) {
    call.transcript.forEach((msg: any, index: number) => {
      if (msg.tool_use) {
        events.push({
          id: `tool-${index}`,
          type: "tool_use",
          timestamp: msg.timestamp,
          title: `Tool: ${msg.tool_use.name}`,
          description: msg.tool_use.result,
        });
      }
    });
  }

  // Call ended
  if (call.status === "completed" || call.status === "failed") {
    events.push({
      id: "end",
      type: call.status === "failed" ? "error" : "call_ended",
      timestamp: call.duration_seconds || call.duration || 0,
      title: call.status === "failed" ? "Call Failed" : "Call Ended",
      description: call.end_reason || (call.status === "completed" ? "Call completed successfully" : "Call ended due to error"),
    });
  }

  return events.sort((a, b) => a.timestamp - b.timestamp);
}

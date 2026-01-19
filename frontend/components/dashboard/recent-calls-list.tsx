"use client";

import React from "react";
import Link from "next/link";
import {
  Phone,
  PhoneCall,
  PhoneOff,
  PhoneMissed,
  ArrowUpRight,
  ArrowDownLeft,
  MoreHorizontal,
  Play,
  FileText,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import { cn, formatDuration, formatRelativeTime, formatPhoneNumber } from "@/lib/utils";

type CallStatus = "in_progress" | "completed" | "failed" | "no_answer" | "busy";
type CallDirection = "inbound" | "outbound";

interface Call {
  id: string;
  from_number: string;
  to_number: string;
  direction: CallDirection;
  status: CallStatus;
  duration_seconds: number;
  agent_id: string;
  agent_name: string;
  created_at: string;
  has_recording?: boolean;
  has_transcript?: boolean;
}

interface RecentCallsListProps {
  calls: Call[];
  isLoading?: boolean;
  showViewAll?: boolean;
  className?: string;
}

const statusConfig: Record<
  CallStatus,
  { label: string; variant: "default" | "success" | "warning" | "destructive" | "secondary"; icon: React.ElementType }
> = {
  in_progress: {
    label: "Live",
    variant: "success",
    icon: PhoneCall,
  },
  completed: {
    label: "Completed",
    variant: "secondary",
    icon: Phone,
  },
  failed: {
    label: "Failed",
    variant: "destructive",
    icon: PhoneOff,
  },
  no_answer: {
    label: "No Answer",
    variant: "warning",
    icon: PhoneMissed,
  },
  busy: {
    label: "Busy",
    variant: "warning",
    icon: PhoneOff,
  },
};

export function RecentCallsList({
  calls,
  isLoading,
  showViewAll = true,
  className,
}: RecentCallsListProps) {
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-32" />
            <Skeleton className="h-8 w-20" />
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div
              key={i}
              className="flex items-center gap-4 p-3 rounded-lg border"
            >
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
            <Phone className="h-5 w-5" />
            Recent Calls
          </CardTitle>
          {showViewAll && (
            <Button variant="ghost" size="sm" asChild>
              <Link href="/calls">View All</Link>
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {calls.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <div className="rounded-full bg-muted p-4 mb-4">
              <Phone className="h-8 w-8 text-muted-foreground" />
            </div>
            <p className="font-medium">No calls yet</p>
            <p className="text-sm text-muted-foreground mt-1">
              Your call history will appear here
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {calls.map((call) => (
              <CallRow key={call.id} call={call} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface CallRowProps {
  call: Call;
}

function CallRow({ call }: CallRowProps) {
  const config = statusConfig[call.status];
  const StatusIcon = config.icon;
  const isLive = call.status === "in_progress";
  const phoneNumber = call.direction === "outbound" ? call.to_number : call.from_number;

  return (
    <div
      className={cn(
        "group flex items-center gap-3 p-3 rounded-lg border transition-all",
        isLive
          ? "border-green-500/30 bg-green-500/5 hover:bg-green-500/10"
          : "hover:border-primary/30 hover:bg-muted/50"
      )}
    >
      <div className="relative">
        <Avatar
          className={cn(
            "h-10 w-10",
            isLive && "ring-2 ring-green-500 ring-offset-2"
          )}
        >
          <AvatarFallback
            className={cn(
              call.status === "in_progress"
                ? "bg-green-100 text-green-600"
                : call.status === "completed"
                ? "bg-gray-100 text-gray-600"
                : "bg-red-100 text-red-600"
            )}
          >
            <StatusIcon className="h-5 w-5" />
          </AvatarFallback>
        </Avatar>
        {isLive && (
          <span className="absolute -bottom-0.5 -right-0.5 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500" />
          </span>
        )}
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium truncate">
            {formatPhoneNumber(phoneNumber)}
          </span>
          {call.direction === "inbound" ? (
            <ArrowDownLeft className="h-3 w-3 text-blue-500" />
          ) : (
            <ArrowUpRight className="h-3 w-3 text-green-500" />
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="truncate">{call.agent_name || "Unknown Agent"}</span>
          <span>•</span>
          <span>{formatRelativeTime(call.created_at)}</span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Badge variant={config.variant} className={cn(isLive && "animate-pulse")}>
          {isLive ? (
            <span className="flex items-center gap-1">
              <span className="relative flex h-1.5 w-1.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
                <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-current" />
              </span>
              Live
            </span>
          ) : call.status === "completed" ? (
            formatDuration(call.duration_seconds)
          ) : (
            config.label
          )}
        </Badge>

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
              <Link href={`/calls/${call.id}`}>View Details</Link>
            </DropdownMenuItem>
            {call.has_recording && (
              <DropdownMenuItem>
                <Play className="mr-2 h-4 w-4" />
                Play Recording
              </DropdownMenuItem>
            )}
            {call.has_transcript && (
              <DropdownMenuItem>
                <FileText className="mr-2 h-4 w-4" />
                View Transcript
              </DropdownMenuItem>
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem asChild>
              <Link href={`/agents/${call.agent_id}`}>View Agent</Link>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}

interface CompactCallListProps {
  calls: Call[];
  isLoading?: boolean;
  className?: string;
}

export function CompactCallList({ calls, isLoading, className }: CompactCallListProps) {
  if (isLoading) {
    return (
      <div className={cn("space-y-2", className)}>
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    );
  }

  return (
    <div className={cn("space-y-1", className)}>
      {calls.map((call) => {
        const config = statusConfig[call.status];
        const StatusIcon = config.icon;
        const phoneNumber = call.direction === "outbound" ? call.to_number : call.from_number;

        return (
          <Link
            key={call.id}
            href={`/calls/${call.id}`}
            className="flex items-center justify-between p-2 rounded-md hover:bg-muted transition-colors"
          >
            <div className="flex items-center gap-2">
              <StatusIcon
                className={cn(
                  "h-4 w-4",
                  call.status === "in_progress"
                    ? "text-green-600"
                    : call.status === "completed"
                    ? "text-muted-foreground"
                    : "text-red-600"
                )}
              />
              <span className="text-sm font-medium">
                {formatPhoneNumber(phoneNumber)}
              </span>
            </div>
            <span className="text-xs text-muted-foreground">
              {formatRelativeTime(call.created_at)}
            </span>
          </Link>
        );
      })}
    </div>
  );
}

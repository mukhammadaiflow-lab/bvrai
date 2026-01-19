"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { Phone, PhoneCall } from "lucide-react";

interface LiveCallIndicatorProps {
  count: number;
  className?: string;
}

export function LiveCallIndicator({ count, className }: LiveCallIndicatorProps) {
  if (count === 0) {
    return (
      <div className={cn("flex items-center gap-2 text-muted-foreground", className)}>
        <div className="relative">
          <Phone className="h-5 w-5" />
        </div>
        <span className="text-sm">No active calls</span>
      </div>
    );
  }

  return (
    <div className={cn("flex items-center gap-3", className)}>
      <div className="relative">
        <div className="absolute inset-0 rounded-full bg-green-500 animate-ping opacity-25" />
        <div className="relative flex h-10 w-10 items-center justify-center rounded-full bg-green-500/10 text-green-600">
          <PhoneCall className="h-5 w-5" />
        </div>
      </div>
      <div>
        <div className="flex items-center gap-2">
          <span className="text-2xl font-bold text-green-600">{count}</span>
          <span className="text-sm font-medium text-muted-foreground">
            {count === 1 ? "call" : "calls"} in progress
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
          </span>
          <span className="text-xs text-green-600 font-medium">Live</span>
        </div>
      </div>
    </div>
  );
}

interface LiveCallBannerProps {
  count: number;
  onViewCalls?: () => void;
  className?: string;
}

export function LiveCallBanner({ count, onViewCalls, className }: LiveCallBannerProps) {
  if (count === 0) return null;

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-lg border border-green-500/30 bg-gradient-to-r from-green-500/10 via-green-500/5 to-transparent p-4",
        className
      )}
    >
      <div className="flex items-center justify-between">
        <LiveCallIndicator count={count} />
        {onViewCalls && (
          <button
            onClick={onViewCalls}
            className="text-sm font-medium text-green-600 hover:text-green-700 transition-colors"
          >
            View calls →
          </button>
        )}
      </div>
      <div className="absolute right-0 top-0 -translate-y-1/4 translate-x-1/4 opacity-10">
        <Phone className="h-32 w-32 text-green-500" />
      </div>
    </div>
  );
}
